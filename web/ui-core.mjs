import init,{ui_core} from './pkg/bonsai_webgpu.js?v=0.4.1.1';
await init({module_or_path:new URL('./pkg/bonsai_webgpu_bg.wasm?v=0.4.1.1',import.meta.url)});
export const call=request=>JSON.parse(ui_core(JSON.stringify(request)));
