import type {ChatMessageTimings} from '$lib/types';

/** Per-request counters, summed once over the selected conversation branch. */
export function speculativeStats(messages: {role?:string;timings?:ChatMessageTimings}[]) {
 let proposed=0,accepted=0,batches=0,seen=false,enabled=false,throttled=false;
 const count=(n:unknown)=>typeof n==='number'&&Number.isFinite(n)?Math.max(0,Math.floor(n)):0;
 for(const message of messages){
  const stats=message.role==='assistant'?message.timings?.ngram:undefined;
  if(!stats)continue;
  seen=true;enabled ||= stats.enabled;throttled ||= !!stats.disabled_for_low_acceptance;
  proposed+=count(stats.proposed);accepted+=Math.min(count(stats.accepted),count(stats.proposed));batches+=count(stats.batches);
 }
 return {seen,enabled,throttled,proposed,accepted,batches,acceptanceRate:proposed?accepted/proposed:null,averageAcceptedLength:batches?accepted/batches:null};
}

export function kvCacheStats(messages: {role?:string;timings?:ChatMessageTimings}[]){
 let total=0,cached=0,currentTotal=0,currentCached=0;
 for(const message of messages){
  const t=message.role==='assistant'?message.timings:undefined;if(!t)continue;
  // Older Bonsai transports reported the full prompt in prompt_n. New records
  // explicitly separate fresh prompt_n from prompt_total_n.
  currentTotal=Math.max(0,t.prompt_total_n??t.prompt_n??0);
  currentCached=Math.min(currentTotal,Math.max(0,t.cache_n??0));
  total+=currentTotal;cached+=currentCached;
 }
 return {total,cached,currentTotal,currentCached};
}
