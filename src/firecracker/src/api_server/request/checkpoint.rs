// Copyright 2025 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

use micro_http::{Method, StatusCode};
use vmm::rpc_interface::VmmAction;

use crate::api_server::parsed_request::{ParsedRequest, RequestError};

pub(crate) fn parse_put_checkpoint(
    request_type_from_path: Option<&str>,
) -> Result<ParsedRequest, RequestError> {
    match request_type_from_path {
        Some(request_type) => match request_type {
            "create" => parse_put_checkpoint_create(),
            "load" => parse_put_checkpoint_load(),
            _ => Err(RequestError::InvalidPathMethod(
                format!("/checkpoint/{}", request_type),
                Method::Put,
            )),
        },
        None => Err(RequestError::Generic(
            StatusCode::BadRequest,
            "Missing checkpoint operation type.".to_string(),
        )),
    }
}

fn parse_put_checkpoint_create() -> Result<ParsedRequest, RequestError> {
    let parsed_request = ParsedRequest::new_sync(VmmAction::CreateCheckpoint);

    Ok(parsed_request)
}

fn parse_put_checkpoint_load() -> Result<ParsedRequest, RequestError> {
    let parsed_request = ParsedRequest::new_sync(VmmAction::LoadCheckpoint);

    Ok(parsed_request)
}

#[cfg(test)]
mod tests {
    use vmm::rpc_interface::VmmAction;

    use crate::api_server::parsed_request::tests::vmm_action_from_request;
    use crate::api_server::request::checkpoint::parse_put_checkpoint;

    #[test]
    fn test_parse_put_checkpoint() {
        assert_eq!(
            vmm_action_from_request(parse_put_checkpoint(Some("create")).unwrap()),
            VmmAction::CreateCheckpoint
        );

        assert_eq!(
            vmm_action_from_request(parse_put_checkpoint(Some("load")).unwrap()),
            VmmAction::LoadCheckpoint
        );

        parse_put_checkpoint(None).unwrap_err();
    }
}
