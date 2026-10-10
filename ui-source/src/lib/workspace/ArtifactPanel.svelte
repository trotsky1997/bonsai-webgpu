<script lang="ts">
 import {workspaceStore as store} from './store.svelte';
 import {decode,htmlPreview,dataUrl} from './preview';
 let {conversationId}:{conversationId:string}=$props();
 let mode=$state('preview'),draft=$state(''),saving=$state(false);
 const files=$derived(store.files(conversationId));const file=$derived(files.find(f=>f.path===store.selectedPath));
 const html=$derived(file&&/\.html?$/i.test(file.path));
 let documentHtml=$state('');
 $effect(()=>{if(file){draft=decode(file);documentHtml=html?htmlPreview(file,files):'';}});
 async function save(){if(!file)return;saving=true;const r=await store.execute('write',{path:file.path,content:draft},conversationId,crypto.randomUUID());store.error=r.isError?r.content:'';saving=false;}
 function download(){if(!file)return;const url=URL.createObjectURL(new Blob([new Uint8Array(file.content??[])],{type:html?'text/html':'application/octet-stream'}));const a=document.createElement('a');a.href=url;a.download=file.path.split('/').at(-1)??'artifact';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
</script>
{#if !store.open}<button class="artifact-toggle" onclick={()=>store.open=true} type="button">产物 {files.length||''}</button>{:else}
<aside class="artifact-panel" aria-label="工作区产物">
 <header><strong>产物</strong><small>浏览器工作区</small><button aria-label="关闭产物面板" onclick={()=>store.open=false} type="button">×</button></header>
 <label class="file-select">文件<select aria-label="选择产物文件" value={store.selectedPath} onchange={e=>store.select(e.currentTarget.value)}><option value="">选择文件</option>{#each files as f}<option value={f.path}>{f.path.replace('/workspace/','')}</option>{/each}</select></label>
 {#if store.error}<p role="alert">{store.error}</p>{/if}
 {#if file}
 <nav aria-label="产物视图"><button class:active={mode==='preview'} onclick={()=>mode='preview'} type="button">预览</button><button class:active={mode==='source'} onclick={()=>mode='source'} type="button">源码</button><button onclick={download} type="button">下载</button><button onclick={()=>window.dispatchEvent(new CustomEvent('bonsai-workspace-intent',{detail:{conversationId,text:'请修改 '+file.path+'：'}}))} type="button">继续修改</button></nav>
 <div class="artifact-content">{#if mode==='source'}<textarea aria-label="产物源码" bind:value={draft} spellcheck="false"></textarea><button class="save" disabled={saving} onclick={save} type="button">{saving?'保存中…':'保存文件'}</button>{:else if html}<iframe title={file.path} sandbox="allow-scripts" referrerpolicy="no-referrer" srcdoc={documentHtml}></iframe>{:else if /\.(png|jpe?g|webp|svg)$/i.test(file.path)}<img alt={file.path} src={dataUrl(file)}/>{:else}<pre>{decode(file)}</pre>{/if}</div>
 <footer>{file.path} · {file.content?.byteLength??0} B</footer>
 {:else}<div class="empty"><h3>让模型生成一个网页</h3><p>也可以处理数据、写报告或修改已有文件。产物会保存在当前对话的工作区。</p></div>{/if}
</aside>{/if}
<style>
 .artifact-toggle{position:fixed;right:12px;top:6px;z-index:35;min-height:44px;padding:8px 14px;border:1px solid var(--border);border-radius:12px;background:var(--background);font-size:12px;}
 .artifact-panel{position:fixed;right:0;top:0;bottom:0;width:min(46vw,640px);z-index:40;display:flex;flex-direction:column;background:var(--background);color:var(--foreground);border-left:1px solid var(--border);box-shadow:-4px 0 24px #0001;min-width:0;}
 header,nav,.file-select{display:flex;align-items:center;gap:8px;padding:8px 12px;flex-wrap:wrap;border-bottom:1px solid var(--border);font-size:12px;}header strong{font-size:16px;}header small{flex:1;opacity:.55;}button{cursor:pointer;min-height:44px;min-width:44px;padding:8px 10px;border-radius:8px;}button:hover,.active{background:var(--muted);}select{min-width:0;flex:1;min-height:44px;background:var(--background);border:1px solid var(--border);border-radius:8px;padding:8px;}.artifact-content{flex:1;min-height:0;overflow:auto;display:flex;flex-direction:column;}iframe{border:0;width:100%;flex:1;min-height:100%;background:white;}textarea{flex:1;min-height:280px;width:100%;resize:none;padding:16px;font:12px/1.6 monospace;background:var(--background);}pre{font:12px/1.6 monospace;white-space:pre-wrap;overflow-wrap:anywhere;padding:16px;}footer{font:10px monospace;padding:8px 12px;opacity:.6;overflow-wrap:anywhere;}img{max-width:100%;height:auto;object-fit:contain;}.empty{padding:24px;font-size:13px;line-height:1.8;opacity:.7;}.save{background:var(--muted);margin:8px;}button:focus-visible,select:focus-visible,textarea:focus-visible{outline:2px solid #087f8c;outline-offset:2px;}@media(max-width:767px){.artifact-panel{width:100%;z-index:60;}.artifact-toggle{top:5px;right:8px;}}@media(prefers-reduced-motion:reduce){*{scroll-behavior:auto;}}
</style>
