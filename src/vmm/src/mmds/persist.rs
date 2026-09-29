// Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//! Defines the structures needed for saving/restoring MmdsNetworkStack.

use std::convert::Infallible;
use std::net::Ipv4Addr;
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};

use super::ns::MmdsNetworkStack;
use crate::mmds::data_store::Mmds;
use crate::snapshot::Persist;
use crate::utils::net::mac::{MAC_ADDR_LEN, MacAddr};

/// State of a MmdsNetworkStack.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MmdsNetworkStackState {
    mac_addr: [u8; MAC_ADDR_LEN as usize],
    ipv4_addr: u32,
    tcp_port: u16,
}

impl<'a> Persist<'a> for MmdsNetworkStack {
    type State = MmdsNetworkStackState;
    type ConstructorArgs = Arc<Mutex<Mmds>>;
    type ApplyArgs = ();
    type Error = Infallible;

    fn save(&self) -> Self::State {
        let mut mac_addr = [0; MAC_ADDR_LEN as usize];
        mac_addr.copy_from_slice(self.mac_addr.get_bytes());

        MmdsNetworkStackState {
            mac_addr,
            ipv4_addr: self.ipv4_addr.into(),
            tcp_port: self.tcp_handler.local_port(),
        }
    }

    fn create(mmds: Self::ConstructorArgs, state: &Self::State) -> Result<Self, Self::Error> {
        Ok(Self::new(
            MacAddr::from_bytes_unchecked(&state.mac_addr),
            Ipv4Addr::from(state.ipv4_addr),
            state.tcp_port,
            mmds,
        ))
    }

    fn restore_in_place(
        &mut self,
        state: &Self::State,
        _: Self::ApplyArgs,
    ) -> Result<(), Self::Error> {
        *self = Self::new(
            MacAddr::from_bytes_unchecked(&state.mac_addr),
            Ipv4Addr::from(state.ipv4_addr),
            state.tcp_port,
            Arc::clone(&self.mmds),
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;
    use crate::mmds::data_store::Mmds;

    #[test]
    fn test_persistence() {
        let ns = MmdsNetworkStack::new(
            MacAddr::from_bytes_unchecked(&[2, 0, 0, 0, 0, 1]),
            Ipv4Addr::new(169, 254, 169, 123),
            8080,
            Arc::new(Mutex::new(Mmds::default())),
        );

        let ns_state = ns.save();
        let serialized_data = bitcode::serialize(&ns_state).unwrap();

        let restored_state = bitcode::deserialize(&serialized_data).unwrap();
        let mmds = Arc::new(Mutex::new(Mmds::default()));
        let mut restored_ns = MmdsNetworkStack::new_with_defaults(None, Arc::clone(&mmds));
        restored_ns.restore_in_place(&restored_state, ()).unwrap();

        assert_eq!(restored_ns.mac_addr, ns.mac_addr);
        assert_eq!(restored_ns.ipv4_addr, ns.ipv4_addr);
        assert_eq!(restored_ns.tcp_handler.local_ipv4_addr(), ns.ipv4_addr);
        assert!(Arc::ptr_eq(&restored_ns.mmds, &mmds));
        assert_eq!(
            restored_ns.tcp_handler.local_port(),
            ns.tcp_handler.local_port()
        );
    }
}
