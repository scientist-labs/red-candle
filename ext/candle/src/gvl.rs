/// GVL (Global VM Lock) release support for Ruby.
///
/// Ruby's GVL prevents other Ruby threads from running while native code
/// executes. For long-running operations (LLM inference, reranking, embedding),
/// we release the GVL so other threads (TUI render loops, HTTP servers, etc.)
/// can run concurrently.
///
/// SAFETY: Code running without the GVL must NOT call any Ruby API.

use std::os::raw::c_void;

type UnblockFn = unsafe extern "C" fn(*mut c_void);

extern "C" {
    fn rb_thread_call_without_gvl(
        func: unsafe extern "C" fn(*mut c_void) -> *mut c_void,
        data1: *mut c_void,
        ubf: Option<UnblockFn>,
        data2: *mut c_void,
    ) -> *mut c_void;
}

/// Run a closure without the GVL. The closure must not call any Ruby API.
pub fn without_gvl<F, R>(f: F) -> R
where
    F: FnOnce() -> R,
{
    struct CallData<F, R> {
        func: Option<F>,
        result: Option<R>,
    }

    unsafe extern "C" fn call_func<F, R>(data: *mut c_void) -> *mut c_void
    where
        F: FnOnce() -> R,
    {
        let data = &mut *(data as *mut CallData<F, R>);
        let func = data.func.take().unwrap();
        data.result = Some(func());
        std::ptr::null_mut()
    }

    let mut data = CallData {
        func: Some(f),
        result: None,
    };

    unsafe {
        rb_thread_call_without_gvl(
            call_func::<F, R>,
            &mut data as *mut _ as *mut c_void,
            None,
            std::ptr::null_mut(),
        );
    }

    data.result.unwrap()
}
