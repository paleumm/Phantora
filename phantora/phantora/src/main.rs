use phantora::args::get_args;
use phantora::simulator;
use phantora::torch_call::{TorchCallJson, TorchCallMsg, NO_KERNEL_OPS};

use cuda_call::{capi, CudaCallMsg, HostId, ResponseId};
use serde::{Deserialize, Serialize};
use std::fs;
use std::os::unix::net::UnixDatagram;

fn main() {
    let env = env_logger::Env::new()
        .filter("PHANTORA_LOG")
        .write_style("PHANTORA_LOG_STYLE");
    env_logger::init_from_env(env);

    let _args = get_args();
    main_loop();
}

#[derive(Serialize, Deserialize)]
enum Message {
    Cuda(CudaCallMsg),
    Torch(TorchCallMsg, i32 /* device index */),
    Exit(ResponseId, i64),
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ExitJson {
    pub pid: u32,
    pub tid: i32,
    pub hostname: String,
    pub cur: i64,
}

/// Called just before the server stops on an invalid program, with the reason.
///
/// A rank blocked waiting for a reply (a synchronize, a communicator init) would
/// otherwise wait forever: a datagram socket does not notice that the server is gone.
/// For every rank thread of this run that is still listening
/// (`<PHANTORA_SOCKET_PREFIX>.node<pid>_t<tid>.sock` accepts a connection), first write
/// the reason to that process's stderr, so it lands in the launcher's log (`run.log`)
/// next to the crash, then send the thread an empty datagram: the rank cannot decode it
/// and aborts, so the launcher exits non-zero instead of hanging. Ranks not blocked
/// abort on their next call, which the stopped server refuses.
fn stop_ranks(reason: &str) {
    let sim_path = std::path::PathBuf::from(capi::simulator_socket_path());
    let (Some(dir), Some(sim_name)) = (sim_path.parent(), sim_path.file_name()) else {
        eprintln!("phantora_server: no socket directory in {}", sim_path.display());
        return;
    };
    let Some(prefix) = sim_name
        .to_str()
        .and_then(|n| n.strip_suffix(".simulator.sock"))
        .map(|p| format!("{p}.node"))
    else {
        eprintln!("phantora_server: unexpected socket name {}", sim_path.display());
        return;
    };
    let entries = match fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(e) => {
            eprintln!("phantora_server: cannot list {}: {e}", dir.display());
            return;
        }
    };
    // (pid, socket) of every rank thread still bound to its socket.
    let mut live: Vec<(u32, UnixDatagram)> = Vec::new();
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        let Some(pid) = name
            .strip_prefix(prefix.as_str())
            .and_then(|rest| rest.strip_suffix(".sock"))
            .and_then(|rest| rest.split_once("_t"))
            .and_then(|(pid, _tid)| pid.parse::<u32>().ok())
        else {
            continue;
        };
        let sock = match UnixDatagram::unbound() {
            Ok(sock) => sock,
            Err(e) => {
                eprintln!("phantora_server: cannot create a socket to stop {name}: {e}");
                continue;
            }
        };
        // Refused for sockets of ranks that already exited: nothing to stop there.
        if sock.connect(entry.path()).is_ok() {
            live.push((pid, sock));
        }
    }
    let mut pids: Vec<u32> = live.iter().map(|(pid, _)| *pid).collect();
    pids.sort_unstable();
    pids.dedup();
    let mut told = 0;
    for pid in &pids {
        let path = format!("/proc/{pid}/fd/2");
        match fs::OpenOptions::new().append(true).open(&path) {
            Ok(mut f) => {
                use std::io::Write;
                if writeln!(f, "\nERROR: phantora simulator stopped this run: {reason}\n").is_ok() {
                    told += 1;
                }
            }
            Err(e) => eprintln!("phantora_server: cannot write to {path}: {e}"),
        }
    }
    let woken = live.iter().filter(|(_, sock)| sock.send(&[]).is_ok()).count();
    eprintln!(
        "phantora_server: stopped; wrote the reason to the stderr of {told} of {} rank \
         process(es) and told {woken} rank thread(s) to abort",
        pids.len()
    );
}

fn main_loop() {
    // Create domain socket for Phantora simulator
    let socket_path = capi::simulator_socket_path();
    let _ = fs::remove_file(&socket_path);
    let recv_socket = UnixDatagram::bind(socket_path).unwrap();

    let mut buf = [0u8; 1024 * 1024];
    let netconfig: netsim::config::Config = netsim::config::read_config(&get_args().net_config);
    println!("netconfig: {:#?}", netconfig);
    let cluster = netsim::config::build_cloud(&netconfig);
    let netsim = netsim::simulator::SimulatorBuilder::new()
        .with_setting(netconfig.simulator)
        .cluster(cluster)
        .host_mapping(netconfig.host_mapping)
        .build()
        .expect("Fail to create network simulator");
    let mut simulator = simulator::Simulator::new(netsim);

    loop {
        let sz = recv_socket.recv(&mut buf).unwrap();
        assert!(sz < buf.len());
        let last = sz - 1;
        let tag = buf[last];
        let buf = &buf[..last];
        let message;
        if tag == 1 {
            // CUDA call message
            let msg = bincode::deserialize::<CudaCallMsg>(&buf).unwrap();
            message = Message::Cuda(msg);
        } else if tag == 2 {
            // torch call message
            let info = std::str::from_utf8(&buf).unwrap();
            let callmsg = serde_json::from_str::<TorchCallJson>(info)
                .unwrap()
                .into_msg();
            if let Some(arg_device) = callmsg.gpu_index() {
                message = Message::Torch(callmsg, arg_device as _);
            } else {
                continue;
            }
        } else if tag == 3 {
            // Exit message
            let info = std::str::from_utf8(&buf).unwrap();
            let exitmsg = serde_json::from_str::<ExitJson>(info).unwrap();
            message = Message::Exit(
                ResponseId {
                    host: HostId {
                        hostname: exitmsg.hostname,
                        pid: exitmsg.pid,
                    },
                    tid: exitmsg.tid,
                },
                exitmsg.cur,
            );
        } else {
            panic!("Unknown message tag {}", tag);
        }

        match message {
            Message::Cuda(msg) => {
                log::debug!("{:?}", msg);
                if let Err(e) = simulator.handle_cuda_call(msg) {
                    // To stderr as well as the log: the log alone is invisible when
                    // PHANTORA_LOG is set below `error`.
                    log::error!("{e}");
                    eprintln!("ERROR: phantora_server: {e}");
                    stop_ranks(&e.to_string());
                    std::process::exit(1);
                }
            }
            Message::Torch(callmsg, arg_device) => {
                log::debug!("{:?}", callmsg);
                let name = callmsg.name.clone();
                let example = if simulator.wants_torch_example(&name) {
                    Some(format!("{:?}", callmsg.args))
                } else {
                    None
                };
                match callmsg.into_call(arg_device) {
                    Some(call) => {
                        simulator.note_simulated_torch_op(&name);
                        simulator.handle_torch_call(call);
                    }
                    // View/metadata ops launch no kernel: dropping them loses nothing.
                    None if NO_KERNEL_OPS.contains(&name.as_str()) => {}
                    None => simulator.note_dropped_torch_op(name, example),
                }
            }
            Message::Exit(host, cur) => {
                log::debug!("{:?} exited at {}", host, cur);
                simulator.handle_exit(host, cur);
            }
        }
        simulator.pump_parked_syncs();
    }
}
