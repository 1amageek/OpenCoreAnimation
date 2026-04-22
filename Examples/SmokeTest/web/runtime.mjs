class SwiftClosureDeallocator {
    constructor(exports) {
        if (typeof FinalizationRegistry === "undefined") {
            throw new Error("The Swift part of JavaScriptKit was configured to require the availability of JavaScript WeakRefs.");
        }
        this.functionRegistry = new FinalizationRegistry((id) => { exports.swjs_free_host_function(id); });
    }
    track(func, func_ref) { this.functionRegistry.register(func, func_ref); }
}
function assertNever(x, message) { throw new Error(message); }
const decode = (kind, payload1, payload2, objectSpace) => {
    switch (kind) {
        case 0: return payload1 === 1;
        case 2: return payload2;
        case 1: case 3: case 7: case 8: return objectSpace.getObject(payload1);
        case 4: return null;
        case 5: return undefined;
        default: assertNever(kind, `JSValue Type kind "${kind}" is not supported`);
    }
};
const decodeArray = (ptr, length, memory, objectSpace) => {
    if (length === 0) return [];
    let result = [];
    for (let index = 0; index < length; index++) {
        const base = ptr + 16 * index;
        result.push(decode(memory.getUint32(base, true), memory.getUint32(base + 4, true), memory.getFloat64(base + 8, true), objectSpace));
    }
    return result;
};
const write = (value, kind_ptr, payload1_ptr, payload2_ptr, is_exception, memory, objectSpace) => {
    memory.setUint32(kind_ptr, writeAndReturnKindBits(value, payload1_ptr, payload2_ptr, is_exception, memory, objectSpace), true);
};
const writeAndReturnKindBits = (value, payload1_ptr, payload2_ptr, is_exception, memory, objectSpace) => {
    const exceptionBit = (is_exception ? 1 : 0) << 31;
    if (value === null) return exceptionBit | 4;
    const writeRef = (kind) => { memory.setUint32(payload1_ptr, objectSpace.retain(value), true); return exceptionBit | kind; };
    const type = typeof value;
    switch (type) {
        case "boolean": memory.setUint32(payload1_ptr, value ? 1 : 0, true); return exceptionBit | 0;
        case "number": memory.setFloat64(payload2_ptr, value, true); return exceptionBit | 2;
        case "string": return writeRef(1);
        case "undefined": return exceptionBit | 5;
        case "object": case "function": return writeRef(3);
        case "symbol": return writeRef(7);
        case "bigint": return writeRef(8);
        default: assertNever(type, `Type "${type}" is not supported yet`);
    }
    throw new Error("Unreachable");
};
let globalVariable;
if (typeof globalThis !== "undefined") globalVariable = globalThis;
else if (typeof window !== "undefined") globalVariable = window;
else if (typeof global !== "undefined") globalVariable = global;
else if (typeof self !== "undefined") globalVariable = self;
class JSObjectSpace {
    constructor() {
        this._heapValueById = new Map();
        this._heapValueById.set(1, globalVariable);
        this._heapEntryByValue = new Map();
        this._heapEntryByValue.set(globalVariable, { id: 1, rc: 1 });
        this._heapNextKey = 2;
    }
    retain(value) {
        const entry = this._heapEntryByValue.get(value);
        if (entry) { entry.rc++; return entry.id; }
        const id = this._heapNextKey++;
        this._heapValueById.set(id, value);
        this._heapEntryByValue.set(value, { id: id, rc: 1 });
        return id;
    }
    retainByRef(ref) { return this.retain(this.getObject(ref)); }
    release(ref) {
        const value = this._heapValueById.get(ref);
        const entry = this._heapEntryByValue.get(value);
        entry.rc--;
        if (entry.rc != 0) return;
        this._heapEntryByValue.delete(value);
        this._heapValueById.delete(ref);
    }
    getObject(ref) {
        const value = this._heapValueById.get(ref);
        if (value === undefined) throw new ReferenceError("Attempted to read invalid reference " + ref);
        return value;
    }
}
class SwiftRuntime {
    constructor(options) {
        this.version = 708;
        this.textDecoder = new TextDecoder("utf-8");
        this.textEncoder = new TextEncoder();
        this._instance = null;
        this.memory = new JSObjectSpace();
        this._closureDeallocator = null;
        this.options = options || {};
        this.getDataView = () => { throw new Error("Please call setInstance() before using any JavaScriptKit APIs."); };
        this.getUint8Array = () => { throw new Error("Please call setInstance() before using any JavaScriptKit APIs."); };
        this.wasmMemory = null;
    }
    setInstance(instance) {
        this._instance = instance;
        const wasmMemory = instance.exports.memory;
        if (wasmMemory instanceof WebAssembly.Memory) {
            let cachedDataView = new DataView(wasmMemory.buffer);
            let cachedUint8Array = new Uint8Array(wasmMemory.buffer);
            this.getDataView = () => { if (cachedDataView.buffer.byteLength === 0) cachedDataView = new DataView(wasmMemory.buffer); return cachedDataView; };
            this.getUint8Array = () => { if (cachedUint8Array.byteLength === 0) cachedUint8Array = new Uint8Array(wasmMemory.buffer); return cachedUint8Array; };
            this.wasmMemory = wasmMemory;
        } else { throw new Error("instance.exports.memory is not a WebAssembly.Memory!?"); }
        if (typeof this.exports._start === "function") throw new Error("JavaScriptKit supports only WASI reactor ABI.");
        if (this.exports.swjs_library_version() != this.version) throw new Error("The versions of JavaScriptKit are incompatible.");
    }
    get instance() { if (!this._instance) throw new Error("WebAssembly instance is not set yet"); return this._instance; }
    get exports() { return this.instance.exports; }
    get closureDeallocator() {
        if (this._closureDeallocator) return this._closureDeallocator;
        if ((this.exports.swjs_library_features() & 1) != 0) this._closureDeallocator = new SwiftClosureDeallocator(this.exports);
        return this._closureDeallocator;
    }
    callHostFunction(host_func_id, line, file, args) {
        const argc = args.length;
        const argv = this.exports.swjs_prepare_host_function_call(argc);
        const dataView = this.getDataView();
        for (let index = 0; index < args.length; index++) {
            const base = argv + 16 * index;
            write(args[index], base, base + 4, base + 8, false, dataView, this.memory);
        }
        let output;
        const callback_func_ref = this.memory.retain((result) => { output = result; });
        this.exports.swjs_call_host_function(host_func_id, argv, argc, callback_func_ref);
        this.exports.swjs_cleanup_host_function_call(argv);
        return output;
    }
    get wasmImports() {
        return {
            swjs_set_prop: (ref, name, kind, payload1, payload2) => { this.memory.getObject(ref)[this.memory.getObject(name)] = decode(kind, payload1, payload2, this.memory); },
            swjs_get_prop: (ref, name, payload1_ptr, payload2_ptr) => writeAndReturnKindBits(this.memory.getObject(ref)[this.memory.getObject(name)], payload1_ptr, payload2_ptr, false, this.getDataView(), this.memory),
            swjs_set_subscript: (ref, index, kind, payload1, payload2) => { this.memory.getObject(ref)[index] = decode(kind, payload1, payload2, this.memory); },
            swjs_get_subscript: (ref, index, payload1_ptr, payload2_ptr) => writeAndReturnKindBits(this.memory.getObject(ref)[index], payload1_ptr, payload2_ptr, false, this.getDataView(), this.memory),
            swjs_encode_string: (ref, bytes_ptr_result) => { const bytes = this.textEncoder.encode(this.memory.getObject(ref)); this.getDataView().setUint32(bytes_ptr_result, this.memory.retain(bytes), true); return bytes.length; },
            swjs_decode_string: (bytes_ptr, length) => this.memory.retain(this.textDecoder.decode(this.getUint8Array().subarray(bytes_ptr, bytes_ptr + length))),
            swjs_load_string: (ref, buffer) => { this.getUint8Array().set(this.memory.getObject(ref), buffer); },
            swjs_call_function: (ref, argv, argc, payload1_ptr, payload2_ptr) => { try { return writeAndReturnKindBits(this.memory.getObject(ref)(...decodeArray(argv, argc, this.getDataView(), this.memory)), payload1_ptr, payload2_ptr, false, this.getDataView(), this.memory); } catch (e) { return writeAndReturnKindBits(e, payload1_ptr, payload2_ptr, true, this.getDataView(), this.memory); } },
            swjs_call_function_no_catch: (ref, argv, argc, payload1_ptr, payload2_ptr) => writeAndReturnKindBits(this.memory.getObject(ref)(...decodeArray(argv, argc, this.getDataView(), this.memory)), payload1_ptr, payload2_ptr, false, this.getDataView(), this.memory),
            swjs_call_function_with_this: (obj_ref, func_ref, argv, argc, payload1_ptr, payload2_ptr) => { try { return writeAndReturnKindBits(this.memory.getObject(func_ref).apply(this.memory.getObject(obj_ref), decodeArray(argv, argc, this.getDataView(), this.memory)), payload1_ptr, payload2_ptr, false, this.getDataView(), this.memory); } catch (e) { return writeAndReturnKindBits(e, payload1_ptr, payload2_ptr, true, this.getDataView(), this.memory); } },
            swjs_call_function_with_this_no_catch: (obj_ref, func_ref, argv, argc, payload1_ptr, payload2_ptr) => writeAndReturnKindBits(this.memory.getObject(func_ref).apply(this.memory.getObject(obj_ref), decodeArray(argv, argc, this.getDataView(), this.memory)), payload1_ptr, payload2_ptr, false, this.getDataView(), this.memory),
            swjs_call_new: (ref, argv, argc) => this.memory.retain(new (this.memory.getObject(ref))(...decodeArray(argv, argc, this.getDataView(), this.memory))),
            swjs_call_throwing_new: (ref, argv, argc, exception_kind_ptr, exception_payload1_ptr, exception_payload2_ptr) => { try { const result = new (this.memory.getObject(ref))(...decodeArray(argv, argc, this.getDataView(), this.memory)); write(null, exception_kind_ptr, exception_payload1_ptr, exception_payload2_ptr, false, this.getDataView(), this.memory); return this.memory.retain(result); } catch (e) { write(e, exception_kind_ptr, exception_payload1_ptr, exception_payload2_ptr, true, this.getDataView(), this.memory); return -1; } },
            swjs_instanceof: (obj_ref, constructor_ref) => this.memory.getObject(obj_ref) instanceof this.memory.getObject(constructor_ref),
            swjs_value_equals: (lhs_ref, rhs_ref) => this.memory.getObject(lhs_ref) == this.memory.getObject(rhs_ref),
            swjs_create_function: (host_func_id, line, file) => { const fileString = this.memory.getObject(file); const func = (...args) => this.callHostFunction(host_func_id, line, fileString, args); const ref = this.memory.retain(func); this.closureDeallocator?.track(func, host_func_id); return ref; },
            swjs_create_typed_array: (constructor_ref, elementsPtr, length) => { const ArrayType = this.memory.getObject(constructor_ref); if (length == 0) return this.memory.retain(new ArrayType()); return this.memory.retain(new ArrayType(this.wasmMemory.buffer, elementsPtr, length).slice()); },
            swjs_create_object: () => this.memory.retain({}),
            swjs_load_typed_array: (ref, buffer) => { this.getUint8Array().set(new Uint8Array(this.memory.getObject(ref).buffer), buffer); },
            swjs_release: (ref) => { this.memory.release(ref); },
            swjs_i64_to_bigint: (value, signed) => this.memory.retain(signed ? value : BigInt.asUintN(64, value)),
            swjs_bigint_to_i64: (ref, signed) => { const obj = this.memory.getObject(ref); if (typeof obj !== "bigint") throw new Error("Expected BigInt"); return signed ? obj : (obj < 0n ? 0n : BigInt.asIntN(64, obj)); },
            swjs_i64_to_bigint_slow: (lower, upper, signed) => { const value = BigInt.asUintN(32, BigInt(lower)) + (BigInt.asUintN(32, BigInt(upper)) << 32n); return this.memory.retain(signed ? BigInt.asIntN(64, value) : value); },
            swjs_unsafe_event_loop_yield: () => { throw new UnsafeEventLoopYield(); },
            swjs_create_oneshot_function: (host_func_id, line, file) => { const fileString = this.memory.getObject(file); return this.memory.retain((...args) => this.callHostFunction(host_func_id, line, fileString, args)); },
            swjs_release_remote: () => {}, swjs_send_job_to_main_thread: () => {}, swjs_listen_message_from_main_thread: () => {},
            swjs_wake_up_worker_thread: () => {}, swjs_listen_message_from_worker_thread: () => {}, swjs_terminate_worker_thread: () => {},
            swjs_get_worker_thread_id: () => -1, swjs_request_sending_object: () => {}, swjs_request_sending_objects: () => {},
        };
    }
}
class UnsafeEventLoopYield extends Error {}
export { SwiftRuntime };