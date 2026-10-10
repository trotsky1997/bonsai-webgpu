<script lang="ts">
 function readConfig(){try{return JSON.parse(localStorage.getItem('bonsai-runtime')??'{}')??{};}catch{return {};}}
 const config=readConfig();
 let context=$state(String(config.context??131072));
 let kv=$state(String(config.kv_bits??4));
 let prefix=$state(config.prefix_cache??true);
 let attention=$state(config.prefill_attention??'flashprefill');
 function apply(){localStorage.setItem('bonsai-runtime',JSON.stringify({context:Number(context),kv_bits:Number(kv),prefill_attention:attention,prefix_cache:prefix}));location.reload();}
</script>
<section class="mb-6 rounded-lg border p-4 space-y-3" aria-label="浏览器推理设置">
 <h3 class="font-medium">浏览器推理</h3>
 <div class="grid gap-3 sm:grid-cols-3">
  <label class="text-sm">上下文
   <select class="block w-full min-h-11 rounded border bg-background px-2" bind:value={context}>
    <option value="4096">4K</option><option value="32768">32K · 模型原配置</option><option value="131072">128K · YaRN ×16 实验</option>
   </select>
  </label>
  <label class="text-sm">KV 缓存
   <select class="block w-full min-h-11 rounded border bg-background px-2" bind:value={kv}><option value="4">INT4 · g32</option><option value="32">FP32 · 质量对照</option></select>
  </label>
  <label class="text-sm">长提示词 attention
   <select class="block w-full min-h-11 rounded border bg-background px-2" bind:value={attention}><option value="dense">完整分块 attention</option><option value="flashprefill">FlashPrefill 思路移植 · 实验</option></select>
  </label>
 </div>
 <label class="flex min-h-11 items-center gap-2 text-sm"><input type="checkbox" bind:checked={prefix}/>固定系统提示词 KV · 自动预热并在刷新后恢复</label>
 <p class="text-sm text-muted-foreground">系统前缀最多保留两份 GPU 快照（合计不超过 128 MiB），本地快照最多 128 MiB。只缓存初始系统块，用户消息和工具结果不写入快照。</p>
 <p class="text-sm text-muted-foreground">KV 按需增长。128K 满载 INT4 KV 约 4.375 GiB；FP32 约 28 GiB。128K 超出模型原始 32K 配置，长文本质量尚需评估。稀疏模式只用于 8K 以上 prefill，decode 保持完整 attention。</p>
 <button class="min-h-11 rounded border px-4" onclick={apply}>应用并重新加载模型</button>
</section>
