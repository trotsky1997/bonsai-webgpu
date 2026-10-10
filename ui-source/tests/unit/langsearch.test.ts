import {describe,it,expect,vi} from 'vitest';
import {searchWeb,LANGSEARCH_ENDPOINT} from '$lib/search/langsearch';
describe('LangSearch',()=>{
 it('sends credentials only to the fixed API and preserves source URLs',async()=>{
  const request=vi.fn(async()=>new Response(JSON.stringify({code:200,data:{webPages:{value:[{name:'Title',url:'https://example.com/doc',text:'Source text'}]}}}))) as unknown as typeof fetch;
  const result=JSON.parse(await searchWeb({query:'test',count:1},'fake-local-key',undefined,request));
  expect(result.results[0].url).toBe('https://example.com/doc');expect(JSON.stringify(result)).not.toContain('fake-local-key');
  expect(vi.mocked(request).mock.calls[0][0]).toBe(LANGSEARCH_ENDPOINT);
  expect(vi.mocked(request).mock.calls[0][1]?.headers).toMatchObject({Authorization:'Bearer fake-local-key'});
 });
 it('rejects invalid inputs before sending a request',async()=>{
  const request=vi.fn() as unknown as typeof fetch;
  for(const args of [{query:''},{query:'x',count:11},{query:'x',includeDomains:['https://evil.com']}])await expect(searchWeb(args,'fake',undefined,request)).rejects.toThrow();
  expect(request).not.toHaveBeenCalled();
 });
 it('does not echo server error bodies or credentials',async()=>{
  const request=vi.fn(async()=>new Response('fake-secret',{status:401})) as unknown as typeof fetch;
  await expect(searchWeb({query:'x'},'fake-secret',undefined,request)).rejects.toThrow('HTTP 401');
 });
 it('bounds results and rejects non-http source links',async()=>{
  const request=vi.fn(async()=>new Response(JSON.stringify({code:200,data:{webPages:{value:[{url:'javascript:alert(1)'},{url:'https://example.com',text:'x'.repeat(9000)}]}}}))) as unknown as typeof fetch;
  const r=JSON.parse(await searchWeb({query:'x'},'fake',undefined,request));expect(r.results).toHaveLength(1);expect(r.results[0].text).toHaveLength(2000);
 });
});

import {extractSearchResults,isWebSearchToolName} from '$lib/utils/search-results';
it('renders source URLs directly from structured LangSearch results',()=>{
 const raw=JSON.stringify({results:[{title:'Official docs',url:'https://docs.langsearch.com/api/web-search-api',text:'Documentation'},{title:'unsafe',url:'javascript:alert(1)'}]});
 expect(isWebSearchToolName('web_search')).toBe(true);
 expect(extractSearchResults(raw)).toEqual([{title:'Official docs',url:'https://docs.langsearch.com/api/web-search-api',highlights:'Documentation',published:undefined}]);
});
