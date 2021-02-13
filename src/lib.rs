#![allow(non_snake_case)]
#![allow(non_camel_case_types)]
#![allow(non_upper_case_globals)]
/*
mod c_api_types {
    include!(concat!(env!("OUT_DIR"), "/c_api_types.rs"));
}

mod c_api {
    use super::ctypes::*;
    include!(concat!(env!("OUT_DIR"), "/c_api.rs"));
}

mod c_api_experimental {
    use super::ctypes::*;
    include!(concat!(env!("OUT_DIR"), "/c_api_experimental.rs"));
}

pub mod core {
    pub use crate::c_api::root::*;
}

pub mod ctypes {
    pub use crate::c_api_types::root::*;
}

pub mod experimental {
    pub use crate::c_api_experimental::root::*;
}

pub use self::core::*;
pub use self::ctypes::*;
*/

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
