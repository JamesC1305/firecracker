// Copyright 2025 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

use micro_http::{Body, Method, StatusCode};
use vmm::{rpc_interface::VmmAction, vmm_config::checkpoint::CreateCheckpointParams};

use crate::api_server::parsed_request::{ParsedRequest, RequestError};

pub(crate) fn parse_put_checkpoint(
    body: Option<&Body>,
    request_type_from_path: Option<&str>,
) -> Result<ParsedRequest, RequestError> {
    match request_type_from_path {
        Some(request_type) => match request_type {
            "create" => parse_put_checkpoint_create(body),
            "load" if body.is_none() => parse_put_checkpoint_load(),
            _ => Err(RequestError::InvalidPathMethod(
                format!("/checkpoint/{}", request_type),
                Method::Put,
            )),
        },
        None => Err(RequestError::Generic(
            StatusCode::BadRequest,
            "Missing reset operation type.".to_string(),
        )),
    }
}

fn parse_put_checkpoint_create(body: Option<&Body>) -> Result<ParsedRequest, RequestError> {
    match body {
        Some(request_body) => {
            let creation_params =
                serde_json::from_slice::<CreateCheckpointParams>(request_body.raw())?;

            let parsed_request =
                ParsedRequest::new_sync(VmmAction::CreateCheckpoint(creation_params));

            Ok(parsed_request)
        }
        None => Err(RequestError::Generic(
            StatusCode::BadRequest,
            "Missing request body".to_string(),
        )),
    }
}

fn parse_put_checkpoint_load() -> Result<ParsedRequest, RequestError> {
    let parsed_request = ParsedRequest::new_sync(VmmAction::LoadCheckpoint);

    Ok(parsed_request)
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use micro_http::Body;
    use vmm::rpc_interface::VmmAction;
    use vmm::vmm_config::checkpoint::CreateCheckpointParams;

    use crate::api_server::parsed_request::tests::vmm_action_from_request;
    use crate::api_server::request::checkpoint::parse_put_checkpoint;

    #[test]
    fn test_parse_put_checkpoint() {
        // 1. Check valid inputs for both endpoints
        let body = r#"{
            "uffd_path": "foo"
        }"#;

        assert_eq!(
            vmm_action_from_request(
                parse_put_checkpoint(Some(&Body::new(body)), Some("create")).unwrap()
            ),
            VmmAction::CreateCheckpoint(CreateCheckpointParams {
                uffd_path: PathBuf::from("foo")
            })
        );

        assert_eq!(
            vmm_action_from_request(parse_put_checkpoint(None, Some("load")).unwrap()),
            VmmAction::LoadCheckpoint
        );

        // 2. Test that not specifying a subpath returns an error
        parse_put_checkpoint(Some(&Body::new(body)), None).unwrap_err();
        parse_put_checkpoint(None, None).unwrap_err();

        // 3. Test that body is required for `create` endpoint
        parse_put_checkpoint(None, Some("create")).unwrap_err();

        // 4. Test that body is ommitted for `load` endpoint
        parse_put_checkpoint(Some(&Body::new(body)), Some("load")).unwrap_err();
    }
}
