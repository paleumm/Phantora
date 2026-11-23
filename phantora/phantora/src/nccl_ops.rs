use cuda_call::{HostId, NcclDatatype};
use netsim::Flow;

/// The flows of one collective, each with its SENDER's rank in the communicator.
pub(crate) type Trace = Vec<(Flow, i32)>;

pub(crate) trait NcclOps {
    fn bcast(
        &mut self,
        root: i32,
        // [(host_id, device_id)]
        ranks: &[(HostId, i32)],
        count: usize,
        dtype: NcclDatatype,
    ) -> Trace;
    // fn reduce(
    //     &mut self,
    //     root: i32,
    //     ranks: &[(u32, i32)],
    //     count: usize,
    //     dtype: NcclDatatype,
    // ) -> Trace;
    fn allreduce(&mut self, ranks: &[(HostId, i32)], count: usize, dtype: NcclDatatype) -> Trace;
    fn allgather(&mut self, ranks: &[(HostId, i32)], count: usize, dtype: NcclDatatype) -> Trace;
    fn reduce_scatter(
        &mut self,
        ranks: &[(HostId, i32)],
        count: usize,
        dtype: NcclDatatype,
    ) -> Trace;
}

pub(crate) struct SimpleRing {
    #[allow(unused)]
    num_rings: u32,
    /// One NIC per GPU: endpoints are "<hostname>/nic<device>" (see flow_endpoints).
    pub(crate) nic_per_gpu: bool,
}

impl Default for SimpleRing {
    fn default() -> Self {
        SimpleRing { num_rings: 1, nic_per_gpu: false }
    }
}

impl SimpleRing {
    #[allow(unused)]
    fn new(num_rings: u32) -> Self {
        SimpleRing { num_rings, nic_per_gpu: false }
    }

    pub(crate) fn with_nic_per_gpu(nic_per_gpu: bool) -> Self {
        SimpleRing { num_rings: 1, nic_per_gpu }
    }

    /// One ring of per-hop flows of `size` bytes. Default: one flow per rank, in
    /// communicator order, between host endpoints (unchanged upstream behaviour).
    ///
    /// nic_per_gpu: the ring visits ranks grouped by host (as NCCL orders it).
    /// Hops inside a host stay single loopback flows (NVLink), exactly as by default.
    /// Each hop between hosts h -> h+1 is split into k channels, k = GPUs per host in
    /// this communicator: channel c carries size/k from host h's c-th local GPU NIC to
    /// host h+1's c-th local GPU NIC, i.e. every channel stays on its own rail
    /// (NCCL rail-aligned rings; PXN relays over NVLink to the rail's GPU).
    fn ring(&self, ranks: &[(HostId, i32)], size: usize) -> Trace {
        let n = ranks.len();
        let mut trace = Trace::new();
        if !self.nic_per_gpu {
            for i in 0..n {
                let src = &ranks[i].0.hostname;
                let dst = &ranks[(i + 1) % n].0.hostname;
                trace.push((Flow::new(size, src, dst, None), i as i32));
            }
            return trace;
        }
        // group comm ranks by host, hosts in order of first appearance
        let mut hosts: Vec<(&str, Vec<usize>)> = Vec::new();
        for (i, (h, _)) in ranks.iter().enumerate() {
            match hosts.iter_mut().find(|(name, _)| *name == h.hostname) {
                Some((_, v)) => v.push(i),
                None => hosts.push((&h.hostname, vec![i])),
            }
        }
        for (_, v) in hosts.iter_mut() {
            v.sort_by_key(|&i| ranks[i].1); // local order = device index = rail
        }
        let order: Vec<usize> = hosts.iter().flat_map(|(_, v)| v.iter().copied()).collect();
        let nh = hosts.len();
        let k = hosts.iter().map(|(_, v)| v.len()).min().unwrap_or(1).max(1);
        for pos in 0..n {
            let a = order[pos];
            let b = order[(pos + 1) % n];
            if nh == 1 || ranks[a].0.hostname == ranks[b].0.hostname {
                let (src, dst) = flow_endpoints(true, (&ranks[a].0, ranks[a].1), (&ranks[b].0, ranks[b].1));
                trace.push((Flow::new(size, &src, &dst, None), a as i32));
            }
        }
        if nh > 1 {
            for h in 0..nh {
                let (sv, dv) = (&hosts[h].1, &hosts[(h + 1) % nh].1);
                for c in 0..k {
                    let bytes = size / k + usize::from(c < size % k);
                    let (s, d) = (sv[c], dv[c]);
                    let (src, dst) = flow_endpoints(true, (&ranks[s].0, ranks[s].1), (&ranks[d].0, ranks[d].1));
                    trace.push((Flow::new(bytes, &src, &dst, None), s as i32));
                }
            }
        }
        trace
    }
}

/// Network endpoints of a flow from (host, device) to (host, device). Default: the
/// hostnames. nic_per_gpu: "<hostname>/nic<device>"; a flow inside one host uses the
/// sender's NIC name at both ends, so the network model treats it as loopback (NVLink).
pub(crate) fn flow_endpoints(
    nic_per_gpu: bool,
    src: (&HostId, i32),
    dst: (&HostId, i32),
) -> (String, String) {
    if !nic_per_gpu {
        return (src.0.hostname.clone(), dst.0.hostname.clone());
    }
    let s = format!("{}/nic{}", src.0.hostname, src.1);
    if src.0.hostname == dst.0.hostname {
        return (s.clone(), s);
    }
    (s, format!("{}/nic{}", dst.0.hostname, dst.1))
}

impl NcclOps for SimpleRing {
    fn bcast(
        &mut self,
        root: i32,
        ranks: &[(HostId, i32)],
        count: usize,
        dtype: NcclDatatype,
    ) -> Trace {
        let mut trace = Trace::new();
        let n = ranks.len();
        let size = count * dtype.size();
        let r = &ranks[root as usize];
        for i in 1..n {
            let d = &ranks[(root as usize + i) % n];
            let (src, dst) = flow_endpoints(self.nic_per_gpu, (&r.0, r.1), (&d.0, d.1));
            trace.push((Flow::new(size, &src, &dst, None), root));
        }
        trace
    }

    fn allreduce(&mut self, ranks: &[(HostId, i32)], count: usize, dtype: NcclDatatype) -> Trace {
        let n = ranks.len();
        self.ring(ranks, count * dtype.size() * 2 * (n - 1) / n)
    }

    fn allgather(&mut self, ranks: &[(HostId, i32)], count: usize, dtype: NcclDatatype) -> Trace {
        let n = ranks.len();
        let send_size = count * dtype.size() * n;
        self.ring(ranks, send_size / n * (n - 1))
    }

    fn reduce_scatter(
        &mut self,
        ranks: &[(HostId, i32)],
        count: usize,
        dtype: NcclDatatype,
    ) -> Trace {
        let n = ranks.len();
        let send_size = count * dtype.size() * n;
        self.ring(ranks, send_size / n * (n - 1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ranks(nhost: usize, ngpu: usize) -> Vec<(HostId, i32)> {
        (0..nhost * ngpu)
            .map(|r| (HostId { hostname: format!("host-{}", r / ngpu + 1), pid: r as u32 }, (r % ngpu) as i32))
            .collect()
    }

    #[test]
    fn default_ring_is_unchanged() {
        let r = ranks(2, 4);
        let t = SimpleRing::default().allreduce(&r, 1 << 20, NcclDatatype::U8);
        assert_eq!(t.len(), 8);
        assert!(t.iter().all(|(f, _)| !f.src.contains('/')));
        assert_eq!(t.iter().filter(|(f, _)| f.src != f.dst).count(), 2);
    }

    #[test]
    fn nic_per_gpu_splits_cross_host_hops_by_rail() {
        let r = ranks(2, 4);
        let size = (1usize << 20) * 2 * 7 / 8;
        let t = SimpleRing::with_nic_per_gpu(true).allreduce(&r, 1 << 20, NcclDatatype::U8);
        let (net, lo): (Vec<_>, Vec<_>) = t.iter().partition(|(f, _)| f.src != f.dst);
        assert_eq!(lo.len(), 6); // 3 NVLink hops per host, full size
        assert!(lo.iter().all(|(f, _)| f.bytes == size));
        assert_eq!(net.len(), 8); // 4 rails x 2 directions
        for (f, _) in &net {
            let rail = |e: &str| e.rsplit_once("/nic").unwrap().1.to_string();
            assert_eq!(rail(&f.src), rail(&f.dst), "{} -> {}", f.src, f.dst);
        }
        let h1: usize = net.iter().filter(|(f, _)| f.src.starts_with("host-1/")).map(|(f, _)| f.bytes).sum();
        assert_eq!(h1, size);
    }

    #[test]
    fn nic_per_gpu_one_gpu_per_host() {
        let r: Vec<_> = (0..4).map(|h| (HostId { hostname: format!("host-{}", h + 1), pid: h }, 2)).collect();
        let t = SimpleRing::with_nic_per_gpu(true).allgather(&r, 1000, NcclDatatype::U8);
        assert_eq!(t.len(), 4);
        assert!(t.iter().all(|(f, _)| f.src.ends_with("/nic2") && f.dst.ends_with("/nic2") && f.src != f.dst));
    }
}
