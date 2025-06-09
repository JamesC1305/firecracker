// Copyright 2025 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

use micro_http::{Method, StatusCode};
use vmm::rpc_interface::VmmAction;

use crate::api_server::parsed_request::{ParsedRequest, RequestError};

pub(crate) fn parse_put_reset(
    request_type_from_path: Option<&str>,
) -> Result<ParsedRequest, RequestError> {
    match request_type_from_path {
        Some(request_type) => match request_type {
            "create" => parse_put_reset_create(),
            "load" => parse_put_reset_load(),
            _ => Err(RequestError::InvalidPathMethod(
                format!("/reset/{}", request_type),
                Method::Put,
            )),
        },
        None => Err(RequestError::Generic(
            StatusCode::BadRequest,
            "Missing reset operation type.".to_string(),
        )),
    }
}

fn parse_put_reset_create() -> Result<ParsedRequest, RequestError> {
    let parsed_request = ParsedRequest::new_sync(VmmAction::CreateResetCheckpoint);

    Ok(parsed_request)
}

fn parse_put_reset_load() -> Result<ParsedRequest, RequestError> {
    let parsed_request = ParsedRequest::new_sync(VmmAction::LoadResetCheckpoint);

    Ok(parsed_request)
}
