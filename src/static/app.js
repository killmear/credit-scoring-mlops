const $ = id => document.getElementById(id);
const key = 'credit-demo-last-request-v1';
const number = x => new Intl.NumberFormat('ru-RU').format(x);
let saved = null;
try { saved = JSON.parse(sessionStorage.getItem(key)); } catch (_) {}
const technical = location.pathname === '/details';
$('home').hidden = technical;
$('technical').hidden = !technical;
$(technical ? 'detailsLink' : 'homeLink').setAttribute('aria-current','page');
if (technical && saved) {
  $('empty').hidden = true; $('trace').hidden = false;
  $('traceTitle').textContent = `Клиент ${saved.client + 1}`;
  $('meta').textContent = `${new Date(saved.time).toLocaleString('ru-RU')} · HTTP ${saved.status} · ${saved.ms} мс (запрос и ожидание ответа)`;
  $('request').textContent = JSON.stringify(saved.request,null,2);
  $('response').textContent = JSON.stringify(saved.response,null,2);
}
function showResult(trace) {
  const p = trace.response.predictions[0];
  $('result').replaceChildren();
  const value = document.createElement('div'); value.className='big';
  value.textContent = (p.probability * 100).toLocaleString('ru-RU',{minimumFractionDigits:2,maximumFractionDigits:2}) + '%';
  const badge = document.createElement('div'); badge.className='badge';
  badge.textContent = p.class ? 'Прогноз: дефолт' : 'Прогноз: без дефолта';
  const note = document.createElement('p'); note.className='muted';
  note.textContent = 'Порог классификации — 50%. Это прогноз модели, а не гарантия исхода.';
  const link = document.createElement('a');link.href='/details';link.textContent='Посмотреть детали этого запроса →';
  $('result').append(value,badge,note,link);
}
if (!technical) (async () => {
  try {
    const response = await fetch('/static/clients.json'); if (!response.ok) throw Error('Не удалось загрузить примеры');
    const clients = await response.json();
    $('client').replaceChildren(...clients.map((_,i) => new Option(`Клиент ${i+1}`,String(i))));
    if (saved && saved.client < clients.length) $('client').value=String(saved.client);
    const select = () => {
      const row=clients[Number($('client').value)];
      $('stats').innerHTML=`<div><span class="muted">Кредитный лимит</span><strong>${number(row.LIMIT_BAL)} TWD</strong></div><div><span class="muted">Возраст</span><strong>${row.AGE} лет</strong></div>`;
      const edu={1:'магистратура / аспирантура',2:'университет',3:'средняя школа',4:'другое'};
      const marriage={1:'в браке',2:'не в браке',3:'другое'};
      $('profile').textContent=`Пол: ${row.SEX===1?'мужской':'женский'}. Образование: ${edu[row.EDUCATION] || 'неизвестно'}. Семейное положение: ${marriage[row.MARRIAGE] || 'неизвестно'}.`;
      $('history').replaceChildren();
      ['Сентябрь','Август','Июль','Июнь','Май','Апрель'].forEach((month,i)=>{
        const pay=row[['PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6'][i]];
        const status=pay>0?`Задержка ${pay} мес.`:({'-2':'Нет потребления','-1':'Оплачено вовремя','0':'Возобновляемый кредит'}[pay]);
        const tr=document.createElement('tr');
        [month,status,number(row[`BILL_AMT${i+1}`]),number(row[`PAY_AMT${i+1}`])].forEach(v=>{const td=document.createElement('td');td.textContent=v;tr.append(td);});
        $('history').append(tr);
      });
      $('result').textContent='Нажмите кнопку расчёта для выбранного клиента.'; $('error').textContent='';
      if (saved && saved.client===Number($('client').value) && saved.status===200) showResult(saved);
    };
    $('client').disabled=false; $('predict').disabled=false; $('client').onchange=select; select();
    $('predict').onclick=async()=>{
      const client=Number($('client').value), request={records:[clients[client]]};
      $('predict').disabled=true; $('client').disabled=true; $('predict').textContent='Считаем…'; $('error').textContent='';
      $('result').textContent='Ожидаем ответ модели…';
      const start=performance.now();
      try {
        const r=await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(request),signal:AbortSignal.timeout(30000)});
        const response=await r.json();
        saved={client,request,response,status:r.status,ms:Math.round(performance.now()-start),time:new Date().toISOString()};
        try { sessionStorage.setItem(key,JSON.stringify(saved)); } catch (_) {}
        if(!r.ok) throw Error(`Сервер вернул HTTP ${r.status}. Подробности — на странице деталей запроса.`);
        showResult(saved);
      } catch(error) { $('error').textContent=error.message; $('result').textContent='Прогноз не получен.'; }
      finally { $('predict').disabled=false; $('client').disabled=false; $('predict').textContent='Рассчитать вероятность дефолта'; }
    };
  } catch(error) { $('error').textContent=error.message; }
})();
