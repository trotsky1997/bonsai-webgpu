<script lang="ts">
 import {searchStore} from './store.svelte';
 let value=$state(searchStore.key),saved=$state(false);
</script>
<section class="mb-6 rounded-lg border p-4 space-y-3" aria-label="LangSearch 搜索设置">
 <h3 class="font-medium">LangSearch 联网搜索</h3>
 <p class="text-sm text-muted-foreground">查询会发送给 LangSearch。Key 仅保存在当前浏览器会话，不进入模型提示词或导出的聊天记录。</p>
 <label class="block text-sm" for="langsearch-key">API Key</label>
 <input id="langsearch-key" type="password" autocomplete="off" spellcheck="false" bind:value oninput={()=>saved=false} class="w-full min-h-11 rounded border bg-background px-3" placeholder="填写自己的 LangSearch Key" />
 <div class="flex flex-wrap items-center gap-3">
  <button class="min-h-11 rounded border px-4" onclick={()=>{searchStore.setKey(value);saved=true;}}>应用到当前会话</button>
  <button class="min-h-11 rounded border px-4" onclick={()=>{value='';searchStore.setKey('');saved=true;}}>清除 Key</button>
  <span class="text-sm" role="status">{saved?'已更新':searchStore.key?'已配置':'未配置'}</span>
 </div>
 <a href="https://langsearch.com/dashboard" target="_blank" rel="noreferrer" class="text-sm underline">获取自己的 Key</a>
</section>
