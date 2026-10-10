use serde_json::{json, Map, Value};
use std::collections::BTreeSet;
type Result<T> = std::result::Result<T, String>;
const MAX_BYTES: usize = 32768;
fn error<T>(s: &str) -> Result<T> { Err(s.into()) }
fn object(v: &Value) -> Result<&Map<String,Value>> { v.as_object().ok_or("Expected an object".into()) }
fn text<'a>(v: &'a Value, key: &str, max: usize) -> Result<&'a str> {
    let s=v[key].as_str().ok_or(format!("{key}: expected text"))?;
    if s.trim().is_empty() || s.len()>max {return error(&format!("{key}: empty or too long"));} Ok(s)
}
fn ident(v: &Value, key: &str) -> Result<String> {
    let s=text(v,key,96)?;
    if !s.bytes().all(|c|c.is_ascii_alphanumeric()||b"_.:-".contains(&c)){return error(&format!("{key}: invalid identifier"));} Ok(s.into())
}
fn array<'a>(v: &'a Value,key:&str,min:usize,max:usize)->Result<&'a Vec<Value>> {
    let a=v[key].as_array().ok_or(format!("{key}: expected array"))?;
    if a.len()<min||a.len()>max{return error(&format!("{key}: requires {min}..{max} items"));} Ok(a)
}
fn finite(v:&Value)->Result<f64>{v.as_f64().filter(|n|n.is_finite()&&n.abs()<=1e15).ok_or("Expected a finite number in [-1e15,1e15]".into())}
fn primitive(v:&Value)->bool{v.is_boolean()||v.is_number()||v.as_str().is_some_and(|s|s.len()<=1024)}
fn unique_items(items:&[Value])->Result<BTreeSet<String>>{
    let mut ids=BTreeSet::new();for item in items{object(item)?;let id=ident(item,"id")?;text(item,"label",256)?;if !ids.insert(id){return error("Duplicate item id");}} Ok(ids)
}
#[derive(Clone)]
enum Field { Bool, Number(f64,f64), Choice(BTreeSet<String>), Text }
impl Field {
 fn accepts(&self,v:&Value)->bool{match self{Self::Bool=>v.is_boolean(),Self::Number(a,b)=>finite(v).is_ok_and(|n|n>=*a&&n<=*b),Self::Choice(c)=>v.as_str().is_some_and(|s|s.is_empty()||c.contains(s)),Self::Text=>v.as_str().is_some_and(|s|s.len()<=256)}}
}
struct Spec { fields:Map<String,Value>, rules:std::collections::BTreeMap<String,Field> }
impl Spec {
 fn add(&mut self,key:String,value:Value,rule:Field)->Result<()> {
  if self.rules.contains_key(&key){return error("Duplicate state field");}
  if !rule.accepts(&value){return error(&format!("Invalid initial state: {key}"));}
  self.fields.insert(key.clone(),value);self.rules.insert(key,rule);Ok(())
 }
 fn state(&self,state:&Value)->Result<Value>{
  let mut out=self.fields.clone();for(k,v)in object(state)?{if !self.rules.get(k).is_some_and(|r|r.accepts(v)){return error(&format!("Invalid state field/value: {k}"));}out.insert(k.clone(),v.clone());}Ok(Value::Object(out))
 }
}
fn spec(def:&Value)->Result<Spec>{
 object(def)?;if def["version"]!=1{return error("Unsupported component version; expected 1");}
 ident(def,"id")?;text(def,"title",256)?;text(def,"summary",1024)?;let data=&def["data"];object(data)?;
 let mut s=Spec{fields:Map::new(),rules:Default::default()};
 match text(def,"type",32)?{
  "comparison"=>{
   let columns=array(data,"columns",1,8)?;let mut keys=BTreeSet::new();for col in columns{let k=ident(col,"key")?;text(col,"label",128)?;if !keys.insert(k){return error("Duplicate column key");}}
   let rows=array(data,"rows",1,64)?;let ids=unique_items(rows)?;
   for row in rows{let values=object(&row["values"])?;for(k,v)in values{if !keys.contains(k)||!primitive(v){return error("Invalid comparison value or column reference");}}}
   s.add("selected".into(),json!(""),Field::Choice(ids))?;s.add("filter".into(),json!(""),Field::Text)?;
  }
  "checklist"=>{
   let items=array(data,"items",1,64)?;let ids=unique_items(items)?;
   for item in items{if let Some(v)=item.get("group"){if !v.as_str().is_some_and(|x|x.len()<=128){return error("Invalid group");}}s.add(format!("checked:{}",item["id"].as_str().unwrap()),item.get("checked").cloned().unwrap_or(json!(false)),Field::Bool)?;}
   if data.get("options").is_some(){let options=array(data,"options",1,16)?;let ids=unique_items(options)?;s.add("option".into(),json!(""),Field::Choice(ids))?;}let _=ids;
  }
  "timeline"=>{let ids=unique_items(array(data,"items",1,64)?)?;s.add("selected".into(),json!(""),Field::Choice(ids))?;}
  "chart"=>{
   if !["line","bar","pie"].contains(&text(data,"kind",8)?){return error("Unknown chart kind");}
   let labels=array(data,"labels",1,64)?;for label in labels{if !label.as_str().is_some_and(|s|!s.is_empty()&&s.len()<=128){return error("Invalid chart label");}}
   let series=array(data,"series",1,8)?;unique_items(series)?;
   for row in series{let values=array(row,"values",labels.len(),labels.len())?;for v in values{let n=finite(v)?;if data["kind"]=="pie"&&n<0.0{return error("Pie values must be nonnegative");}}s.add(format!("visible:{}",row["id"].as_str().unwrap()),json!(true),Field::Bool)?;}
  }
  "parameters"=>{
   let controls=array(data,"controls",1,12)?;unique_items(controls)?;
   for c in controls{
    let key=format!("param:{}",c["id"].as_str().unwrap());
    match text(c,"kind",16)?{
     "slider"|"number"=>{let min=c.get("min").map(finite).transpose()?.unwrap_or(0.0);let max=c.get("max").map(finite).transpose()?.unwrap_or(100.0);if min>=max{return error("Control min must be less than max");}if let Some(step)=c.get("step"){if finite(step)?<=0.0{return error("Control step must be positive");}}let value=c.get("value").cloned().unwrap_or(json!(min));s.add(key,value,Field::Number(min,max))?;}
     "switch"=>{s.add(key,c.get("value").cloned().unwrap_or(json!(false)),Field::Bool)?;}
     "select"=>{let options=array(c,"options",1,16)?;let mut values=BTreeSet::new();for o in options{text(o,"label",128)?;let value=text(o,"value",128)?.to_string();if !values.insert(value){return error("Duplicate select value");}}let initial=c.get("value").cloned().unwrap_or(options[0]["value"].clone());s.add(key,initial,Field::Choice(values))?;}
     _=>return error("Unknown control kind")
    }
   }
   if data.get("outputs").is_some(){let outputs=array(data,"outputs",1,8)?;unique_items(outputs)?;for o in outputs{let mut budget=4096;check_expr(&o["expr"],&s,0,&mut budget,false)?;}}
   if let Some(chart)=data.get("chart"){if !["line","bar","pie"].contains(&text(chart,"kind",8)?){return error("Unknown calculated chart kind");}let series=array(chart,"series",1,4)?;unique_items(series)?;for r in series{let mut budget=4096;check_expr(&r["expr"],&s,0,&mut budget,false)?;}if let Some(labels)=chart.get("labels"){let labels=labels.as_array().ok_or("Chart labels must be an array")?;if labels.len()>256||labels.iter().any(|x|!x.as_str().is_some_and(|s|s.len()<=128)){return error("Invalid calculated chart labels");}}}
  }
  "diagram"=>{
   let nodes=array(data,"nodes",1,32)?;let ids=unique_items(nodes)?;
   for n in nodes{for key in ["x","y"]{let value=finite(&n[key])?;if !(0.0..=100.0).contains(&value){return error("Node coordinates must be in 0..100");}}}
   for edge in array(data,"edges",0,64)?{let from=text(edge,"from",96)?;let to=text(edge,"to",96)?;if !ids.contains(from)||!ids.contains(to){return error("Unknown diagram node reference");}}
   s.add("selected".into(),json!(""),Field::Choice(ids))?;
  }
  _=>return error("Unknown component type")
 }
 Ok(s)
}
fn check_expr(v:&Value,s:&Spec,depth:usize,budget:&mut usize,in_map:bool)->Result<()>{
 if depth>16||*budget==0{return error("Expression complexity limit");}*budget-=1;
 if primitive(v){if v.is_number(){finite(v)?;}return Ok(());}if let Some(a)=v.as_array(){if a.len()>256{return error("Sequence limit 256");}for x in a{check_expr(x,s,depth+1,budget,in_map)?;}return Ok(());}
 object(v)?;if let Some(p)=v["param"].as_str(){if !s.rules.contains_key(&format!("param:{p}")){return error("Unknown parameter reference");}return Ok(());}
 if v["var"]=="x"&&in_map{return Ok(());}
 let op=text(v,"op",16)?;let a=array(v,"args",1,8)?;
 let n=match op{"add"|"mul"|"min"|"max"=>1..=8,"sub"|"div"|"pow"|"lt"|"le"|"eq"|"map"=>2..=2,"if"|"range"=>3..=3,"sum"|"avg"|"abs"|"round"=>1..=1,_=>return error("Unknown calculation operation")};
 if !n.contains(&a.len()){return error("Wrong operation arity");}
 for (i,x)in a.iter().enumerate(){check_expr(x,s,depth+1,budget,in_map||(op=="map"&&i==1))?;}Ok(())
}
fn eval(v:&Value,state:&Value,x:Option<&Value>,depth:usize,budget:&mut usize)->Result<Value>{
 if depth>16||*budget==0{return error("Calculation budget exceeded");}*budget-=1;
 if primitive(v){if v.is_number(){finite(v)?;}return Ok(v.clone());}if let Some(a)=v.as_array(){if a.len()>256{return error("Sequence limit 256");}return a.iter().map(|v|eval(v,state,x,depth+1,budget)).collect::<Result<Vec<_>>>().map(Value::Array);}
 if let Some(p)=v["param"].as_str(){return state.get(format!("param:{p}")).cloned().ok_or("Unknown parameter".into());}
 if v["var"]=="x"{return x.cloned().ok_or("Variable outside map".into());}
 let op=text(v,"op",16)?;let a=v["args"].as_array().ok_or("Missing args")?;
 if op=="if"{let condition=eval(&a[0],state,x,depth+1,budget)?.as_bool().ok_or("if condition must be boolean")?;return eval(&a[if condition{1}else{2}],state,x,depth+1,budget);}
 if op=="map"{let values=eval(&a[0],state,x,depth+1,budget)?;let values=values.as_array().ok_or("map requires a sequence")?;if values.len()>256{return error("Sequence limit 256");}return values.iter().map(|x|eval(&a[1],state,Some(x),depth+1,budget)).collect::<Result<Vec<_>>>().map(Value::Array);}
 let values=a.iter().map(|v|eval(v,state,x,depth+1,budget)).collect::<Result<Vec<_>>>()?;
 if op=="eq"{return Ok(json!(values[0]==values[1]));}
 if op=="range"{let start=finite(&values[0])?;let end=finite(&values[1])?;let step=finite(&values[2])?;if step==0.0||(end-start)*step<0.0{return error("Invalid range direction or step");}let n=((end-start)/step).floor()+1.0;if !(0.0..=256.0).contains(&n){return error("Sequence limit 256");}return Ok(Value::Array((0..n as usize).map(|i|json!(start+step*i as f64)).collect()));}
 let numbers=if ["sum","avg"].contains(&op){values[0].as_array().ok_or("Aggregate requires a sequence")?.iter().map(finite).collect::<Result<Vec<_>>>()?}else{values.iter().map(finite).collect::<Result<Vec<_>>>()?};
 if op=="lt"||op=="le"{return Ok(json!(if op=="lt"{numbers[0]<numbers[1]}else{numbers[0]<=numbers[1]}));}
 let result=match op{"add"|"sum"=>numbers.iter().sum(),"mul"=>numbers.iter().product(),"sub"=>numbers[0]-numbers[1],"div"=>{if numbers[1]==0.0{return error("Division by zero");}numbers[0]/numbers[1]},"pow"=>numbers[0].powf(numbers[1]),"min"=>numbers.iter().copied().fold(f64::INFINITY,f64::min),"max"=>numbers.iter().copied().fold(f64::NEG_INFINITY,f64::max),"avg"=>{if numbers.is_empty(){return error("Empty average");}numbers.iter().sum::<f64>()/numbers.len()as f64},"abs"=>numbers[0].abs(),"round"=>numbers[0].round(),_=>return error("Unknown operation")};
 if !result.is_finite()||result.abs()>1e15{return error("Calculation result outside bounds");}Ok(json!(result))
}
fn evaluate(def:&Value,state:&Value)->Result<Value>{
 let mut out=Vec::new();let mut budget=4096;
 if let Some(outputs)=def["data"]["outputs"].as_array(){for o in outputs{out.push(json!({"id":o["id"],"label":o["label"],"unit":o.get("unit").unwrap_or(&Value::Null),"value":eval(&o["expr"],state,None,0,&mut budget)?}));}}
 let chart=if let Some(chart)=def["data"].get("chart"){let mut series=Vec::new();let mut len=None;for r in chart["series"].as_array().unwrap(){let values=eval(&r["expr"],state,None,0,&mut budget)?;let a=values.as_array().ok_or("Chart expression must produce an array")?;if a.is_empty()||a.len()>256||len.is_some_and(|l|l!=a.len()){return error("Invalid calculated chart series length");}for v in a{if finite(v)?<0.0&&chart["kind"]=="pie"{return error("Negative pie value");}}len=Some(a.len());series.push(json!({"id":r["id"],"label":r["label"],"values":values}));}let n=len.unwrap_or(0);let labels=chart.get("labels").cloned().unwrap_or(json!((1..=n).map(|i|i.to_string()).collect::<Vec<_>>()));if labels.as_array().map(|a|a.len())!=Some(n){return error("Chart labels/values length mismatch");}json!({"kind":chart["kind"],"labels":labels,"series":series})}else{Value::Null};
 Ok(json!({"outputs":out,"chart":chart}))
}
pub fn dispatch(input:&str)->Result<String>{
 if input.len()>MAX_BYTES*3{return error("Frame exceeds 96 KiB");}let req:Value=serde_json::from_str(input).map_err(|e|e.to_string())?;let def=&req["definition"];
 if def.to_string().len()>MAX_BYTES{return error("Component exceeds 32 KiB");}let s=spec(def)?;
 let response=match req["op"].as_str(){
  Some("validate")=>{let state=s.state(req.get("state").unwrap_or(&json!({})))?;evaluate(def,&state)?;json!({"definition":def,"state":state,"fields":s.rules.keys().collect::<Vec<_>>()})},
  Some("evaluate")=>{let state=s.state(&req["state"])?;evaluate(def,&state)?},
  Some("merge")=>{let current=s.state(&req["current"])?;let base=s.state(&req["base"])?;let patch=object(&req["patch"])?;let mut conflicts=Vec::new();let mut merged=current.clone();for(k,v)in patch{if !s.rules.get(k).is_some_and(|r|r.accepts(v)){return error(&format!("Invalid patch field/value: {k}"));}if current[k]!=base[k]&&current[k]!=*v{conflicts.push(k);}merged[k]=v.clone();}if !conflicts.is_empty(){json!({"status":"conflict","fields":conflicts,"state":current})}else{json!({"status":"ok","state":merged})}},
  _=>return error("Unknown UI core operation")
 };serde_json::to_string(&response).map_err(|e|e.to_string())
}
#[cfg(test)]
mod tests{
 use super::*;
 fn def()->Value{json!({"version":1,"id":"budget","type":"parameters","title":"Budget","summary":"Calculate savings","data":{"controls":[{"id":"monthly","label":"Monthly","kind":"slider","min":0,"max":10000,"value":100},{"id":"enabled","label":"Enabled","kind":"switch","value":true}],"outputs":[{"id":"annual","label":"Annual","expr":{"op":"if","args":[{"param":"enabled"},{"op":"mul","args":[{"param":"monthly"},12]},0]}}],"chart":{"kind":"line","series":[{"id":"total","label":"Total","expr":{"op":"map","args":[{"op":"range","args":[1,12,1]},{"op":"mul","args":[{"var":"x"},{"param":"monthly"}]}]}}]}}})}
 #[test]fn calculations(){let d=def();let r:Value=serde_json::from_str(&dispatch(&json!({"op":"validate","definition":d}).to_string()).unwrap()).unwrap();let e=evaluate(&d,&r["state"]).unwrap();assert_eq!(e["outputs"][0]["value"],1200.0);assert_eq!(e["chart"]["series"][0]["values"][11],1200.0);}
 #[test]fn merge_fields_and_conflicts(){let d=def();let base=json!({"param:monthly":100,"param:enabled":true});let current=json!({"param:monthly":200,"param:enabled":true});let merged:Value=serde_json::from_str(&dispatch(&json!({"op":"merge","definition":d,"base":base,"current":current,"patch":{"param:enabled":false}}).to_string()).unwrap()).unwrap();assert_eq!(merged["status"],"ok");assert_eq!(merged["state"]["param:monthly"],200);let conflict:Value=serde_json::from_str(&dispatch(&json!({"op":"merge","definition":d,"base":base,"current":current,"patch":{"param:monthly":300,"param:enabled":false}}).to_string()).unwrap()).unwrap();assert_eq!(conflict["status"],"conflict");assert_eq!(conflict["state"]["param:enabled"],true);}
 #[test]fn bounded_calculation(){let d=def();let s=spec(&d).unwrap();let mut budget=4096;let expr=json!({"op":"range","args":[0,1000000,1]});check_expr(&expr,&s,0,&mut budget,false).unwrap();assert!(eval(&expr,&json!({}),None,0,&mut budget).is_err());assert!(check_expr(&json!({"op":"fetch","args":["https://example.com"]}),&s,0,&mut budget,false).is_err());assert!(dispatch(&json!({"op":"merge","definition":d,"base":{},"current":{},"patch":{"__proto__":{}}}).to_string()).is_err());}
 #[test]fn rejects_unknown_version(){let mut d=def();d["version"]=json!(2);assert!(spec(&d).is_err());}
}
