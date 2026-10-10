/** @param {string} id @param {import("../ui-source/src/lib/interactive/types").UiKind} type @param {Record<string,any>} data @returns {import("../ui-source/src/lib/interactive/types").UiDefinition} */
const base=(id,type,data)=>({version:1,id,type,title:'交互验收 · '+type,summary:'用于组件、状态与计算验收的合成样例',data});
export const fixtures=[
 base('compare','comparison',{columns:[{key:'price',label:'价格'},{key:'weight',label:'重量'}],rows:[{id:'a',label:'轻便方案',values:{price:120,weight:'1 kg'}},{id:'b',label:'舒适方案',values:{price:200,weight:'2 kg'}}]}),
 base('camp','checklist',{items:[{id:'tent',label:'帐篷',group:'住宿',description:'确认配件完整'},{id:'water',label:'饮用水',group:'补给'},{id:'food',label:'食物',group:'补给'}],options:[{id:'light',label:'轻装'},{id:'comfort',label:'舒适'}]}),
 base('plan','timeline',{items:[{id:'p1',label:'准备',date:'第一天',description:'整理清单与路线'},{id:'p2',label:'出发',date:'第二天',description:'检查天气'}]}),
 base('chart','chart',{kind:'bar',labels:['甲','乙','丙'],series:[{id:'a',label:'耗时',values:[4,2,3]},{id:'b',label:'成本',values:[2,3,1]}]}),
 base('budget','parameters',{controls:[{id:'monthly',label:'每月投入',kind:'slider',min:0,max:1000,value:100,step:10},{id:'enabled',label:'启用',kind:'switch',value:true},{id:'mode',label:'频率',kind:'select',value:'a',options:[{value:'a',label:'每月'},{value:'b',label:'每季'}]}],outputs:[{id:'annual',label:'年度合计',unit:'元',expr:{op:'if',args:[{param:'enabled'},{op:'mul',args:[{param:'monthly'},12]},0]}}],chart:{kind:'line',series:[{id:'total',label:'累计',expr:{op:'map',args:[{op:'range',args:[1,12,1]},{op:'mul',args:[{var:'x'},{param:'monthly'}]}]}}]}}),
 base('flow','diagram',{nodes:[{id:'input',label:'输入',x:10,y:50,description:'用户的文本消息'},{id:'gpu',label:'WebGPU',x:50,y:50,description:'本地推理'},{id:'output',label:'输出',x:90,y:50,description:'流式回复'}],edges:[{from:'input',to:'gpu',label:'Token'},{from:'gpu',to:'output',label:'文本'}]})
];
