import {browser} from '$app/environment';
import type {OpenAIToolDefinition} from '$lib/types';
import {searchWeb} from './langsearch';
const storage='bonsai-langsearch-session-key';
class SearchStore {
 key=$state(browser?sessionStorage.getItem(storage)??'':'');
 setKey(value:string){this.key=value.trim();if(this.key)sessionStorage.setItem(storage,this.key);else sessionStorage.removeItem(storage);}
 get tools():OpenAIToolDefinition[]{return this.key?[{type:'function',function:{name:'web_search',description:'Search the live web with LangSearch. Returns webpage text and source URLs. Use for current information; cite URLs in your answer. Query is sent to LangSearch.',parameters:{type:'object',properties:{query:{type:'string'},count:{type:'integer',description:'1–10 results, default 5'},freshness:{type:'string',description:'noLimit, oneDay, oneWeek, oneMonth, oneYear, YYYY-MM-DD or YYYY-MM-DD..YYYY-MM-DD'},includeDomains:{type:'array',items:{type:'string'}},excludeDomains:{type:'array',items:{type:'string'}}},required:['query']}}}]:[];}
 async execute(args:Record<string,unknown>,signal?:AbortSignal){try{return {content:await searchWeb(args,this.key,signal),isError:false};}catch(e){if(signal?.aborted)throw e;return {content:String((e as Error).message),isError:true};}}
}
export const searchStore=new SearchStore();
