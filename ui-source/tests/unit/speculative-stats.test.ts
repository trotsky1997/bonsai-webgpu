import {expect,it} from 'vitest';
import {speculativeStats,kvCacheStats} from '$lib/utils/speculative-stats';
it('weights acceptance by proposals and averages contiguous prefixes over all batches',()=>{
 const message=(proposed:number,accepted:number,batches:number)=>({role:'assistant',timings:{ngram:{enabled:true,proposed,accepted,batches}}});
 const value=speculativeStats([message(4,4,1),message(12,4,3),{...message(99,99,99),role:'user'}]);
 expect(value.acceptanceRate).toBe(.5);expect(value.averageAcceptedLength).toBe(2);expect(value.accepted).toBe(8);expect(value.batches).toBe(4);
});
it('computes KV rates from token totals and handles older Bonsai full-prompt counters',()=>{
 const stats=kvCacheStats([{role:'assistant',timings:{prompt_n:100,cache_n:80}},{role:'assistant',timings:{prompt_n:50,prompt_total_n:200,cache_n:150}},{role:'user',timings:{prompt_n:999,cache_n:999}}]);
 expect(stats).toEqual({total:300,cached:230,currentTotal:200,currentCached:150});
 expect(kvCacheStats([]).total).toBe(0);
});
it('distinguishes absent stats, no proposals, disabled and fully rejected drafts',()=>{
 expect(speculativeStats([]).seen).toBe(false);
 expect(speculativeStats([{role:'assistant',timings:{ngram:{enabled:true,proposed:0,accepted:0,batches:0}}}])).toMatchObject({seen:true,enabled:true,acceptanceRate:null,averageAcceptedLength:null});
 expect(speculativeStats([{role:'assistant',timings:{ngram:{enabled:false,proposed:0,accepted:0,batches:0}}}]).enabled).toBe(false);
 expect(speculativeStats([{role:'assistant',timings:{ngram:{enabled:true,proposed:8,accepted:0,batches:2}}}])).toMatchObject({acceptanceRate:0,averageAcceptedLength:0});
});
