// Additional case scenarios share the same evidence, review and clip workflow.
const scenarioOptions=[
 ['helmet','Без каски','Явное распознавание головы без каски. Невидимая голова — «неизвестно».'],
 ['proximity','Сближение с техникой','Движущийся транспорт или размеченная область оборудования.'],
 ['phone','Возможное использование телефона','Телефон рядом с измеренной рукой, подтверждённый несколькими наблюдениями.'],
 ['fire','Открытый огонь','Вне отмеченных разрешённых областей.'],
 ['smoking','Возможное курение','Видимый признак рядом с человеком; на дальних планах может не хватать деталей. Требуется проверка клипа.'],
 ['harness','Работа на высоте без привязи','Нужна зона «Работа на высоте» и явный класс No Harness.'],
 ['load_posture','Переноска с сильным наклоном · эксперимент','Пока только распознанные сумки/чемоданы. Массу и коробки не определяет.'],
 ['smoke','Дым','Возможный дым: отличие от пыли и пара не подтверждено. Оператор проверяет клип.'],
 ['vest','Без сигнального жилета','Дополнительный контроль СИЗ по явному отрицательному классу.'],
 ['mask','Без защиты лица','Класс Mask; тип и сертификация респиратора не определяются.']
];
const extraLabels={no_helmet:'Без каски',no_vest:'Без сигнального жилета',no_mask:'Без защиты лица',no_harness_at_height:'Без привязи в зоне высоты',equipment_proximity:'Опасное сближение с техникой',possible_phone_use:'Возможное использование телефона',fire:'Открытый огонь',smoke:'Дым',possible_smoking:'Возможное курение',barrier_crossing:'Вход за ограждение',under_load:'Человек под грузом',under_conveyor:'Человек под конвейером',unsafe_load_posture:'Наклон при переноске · эксперимент'};
Object.assign(labels,extraLabels);
const zoneKinds={danger_zone:'Опасная зона',barrier:'За ограждением',under_load:'Под подвешенным грузом',under_conveyor:'Под конвейером',height:'Работа на высоте',equipment:'Работающее оборудование',smoking_allowed:'Курение разрешено',fire_allowed:'Огонь разрешён',reflection:'Зеркало · исключить отражение по положению ног'};
function extraScenarioSelected(prefix='extra'){return scenarioOptions.some(([key])=>$(prefix+'-'+key)?.checked)}
function scenarioValues(prefix){const c={};for(const [key] of scenarioOptions)c[key+'_detection']=$(prefix+'-'+key).checked;c.distant_people=$(prefix+'-distant_people')?.checked??true;c.phone_work_context_only=$(prefix+'-phone_work_context_only')?.checked??false;for(const [key] of scenarioParams)c[key]=Number($(prefix+'-'+key).value);return c}
function setScenarioValues(prefix,c={}){for(const [key] of scenarioOptions)$(prefix+'-'+key).checked=Boolean(c[key+'_detection']);$(prefix+'-distant_people').checked=c.distant_people??true;$(prefix+'-phone_work_context_only').checked=c.phone_work_context_only??false;for(const [key,,value] of scenarioParams)$(prefix+'-'+key).value=c[key]??value;updateSmokeExplanation(prefix)}
const scenarioParams=[['phone_confidence','Порог телефона',.3,.15,.95,.05],['ppe_hold','Подтверждение СИЗ, с',1.5,.3,15,.1],['object_hold','Подтверждение объектов, с',1,.3,15,.1],['object_confidence','Порог объектов',.5,.15,.95,.05],['proximity_heights','Дистанция без калибровки, ростов',.6,.1,5,.1],['proximity_m','Дистанция с калибровкой, м',3,.3,15,.1]];
const scenarioParamHelp={
 phone_confidence:'Ниже порог — больше кандидатов телефона, но больше ложных тревог. Выше — строже отбор, но можно пропустить маленький телефон. Значение 0,30 не означает точность 30%.',
 ppe_hold:'Минимальное время устойчивых свежих признаков отсутствия СИЗ. Короткая вспышка отбрасывается; нужны минимум три разных наблюдения. Обработка видео добавляет задержку.',
 object_hold:'Минимальная длительность устойчивого признака телефона, огня, дыма и других объектов. Увеличение отбрасывает короткие срабатывания, но задерживает предупреждение.',
 object_confidence:'Порог оценки детектора объектов. Выше — меньше слабых кандидатов и больше возможных пропусков. Это не измеренная точность модели на ваших камерах.',
 proximity_heights:'Расстояние на изображении относительно видимого роста человека. Это не метры; ракурс и перспектива могут искажать оценку.',
 proximity_m:'Порог расстояния в метрах применяется при калибровке пола. Без четырёх измеренных точек используется относительное расстояние.'
};
function controls(prefix){
 return `<div class="scenario-extension"><div class="between"><h3>Сценарии из задания и СИЗ</h3><span class="pill amber">Нужна валидация</span></div><p class="hint">Включайте нужные модели: каждая увеличивает время анализа. Новые сценарии пока не имеют независимых метрик.</p><div class="extra-scenario-grid"><label class="extra-scenario"><input type="checkbox" id="${prefix}-distant_people" checked><span><b>Дальняя камера</b><small>Более подробный поиск людей в широких кадрах. Исходное разрешение сохраняется для анализа.</small></span></label><label class="extra-scenario"><input type="checkbox" id="${prefix}-phone_work_context_only"><span><b>Телефон: только при движении или работе</b><small>Дополнительное ограничение. По умолчанию проверяются и неподвижные люди.</small></span></label>${scenarioOptions.map(([key,title,desc])=>`<label class="extra-scenario"><input type="checkbox" id="${prefix}-${key}"><span><b>${title}</b><small>${desc}</small></span></label>`).join('')}</div><aside id="${prefix}-smoke-note" class="smoke-uncertainty" hidden><b>Дым, пыль или пар?</b><p>Модель может спутать эти признаки. Подтверждение во времени убирает краткие всплески, но не доказывает, что это дым. Проверьте источник облака и соседние кадры; событие остаётся на проверке оператора.</p><p>Надёжное разделение дыма, пыли и пара на заводских камерах требует размеченных примеров и отдельной оценки. Текущие метрики такого различения не подтверждены.</p></aside><p class="hint">Ограждения, область под грузом и под конвейером задаются типом зоны на кадре. Объект и высоту физической конструкции система самостоятельно не подтверждает.</p><details><summary>Чувствительность новых сценариев</summary><div class="parameter-explainer"><p><b>Порог уверенности</b> отбирает кандидатов модели: ниже — больше находок и ложных тревог, выше — больше пропусков. Это оценка модели, а не процент её точности.</p><p><b>Время подтверждения</b> — сколько должен сохраняться признак до предупреждения. Большая выдержка уменьшает короткие срабатывания, но увеличивает задержку. Отсутствие видимого признака не доказывает отсутствие нарушения.</p></div><div class="form-grid">${scenarioParams.map(([key,title,value,min,max,step])=>`<label>${title}<input id="${prefix}-${key}" type="number" value="${value}" min="${min}" max="${max}" step="${step}" aria-describedby="${prefix}-${key}-help"><small id="${prefix}-${key}-help" class="parameter-help">${scenarioParamHelp[key]}</small></label>`).join('')}</div></details></div>`
}
function updateSmokeExplanation(prefix){const note=$(prefix+'-smoke-note');if(note)note.hidden=!$(prefix+'-smoke')?.checked}

$('step1').querySelector('.scenarios').insertAdjacentHTML('afterend',controls('extra'));
$('live').querySelector('.live-checks').insertAdjacentHTML('afterend','<details class="live-extra-options"><summary>Дополнительные сценарии и СИЗ</summary>'+controls('live-extra')+'</details>');
for(const prefix of ['extra','live-extra'])$(prefix+'-smoke').addEventListener('change',()=>updateSmokeExplanation(prefix));
for(const [id] of [['zone-name'],['live-zone-name']])$(id).closest('label').insertAdjacentHTML('beforebegin',`<label>Назначение зоны<select id="${id==='zone-name'?'zone-kind':'live-zone-kind'}">${Object.entries(zoneKinds).map(([k,v])=>`<option value="${k}">${v}</option>`).join('')}</select></label>`);
for(const id of ['filter-kind','clip-kind'])$(id).insertAdjacentHTML('beforeend',Object.entries(extraLabels).map(([k,v])=>`<option value="${k}">${v}</option>`).join(''));
const applyBaseConfig=applyConfig;
applyConfig=function(c={}){applyBaseConfig(c);setScenarioValues('extra',c)};
const baseConfigValues=config;
config=function(){return {...baseConfigValues(),...scenarioValues('extra')}};
const baseLiveConfig=liveConfig;
liveConfig=function(){return {...baseLiveConfig(),...scenarioValues('live-extra')}};
const baseReview=renderReview;
renderReview=function(){baseReview();const enabled=scenarioOptions.filter(([key])=>$('extra-'+key).checked).map(([,title])=>title);const row=Array.from($('review').querySelectorAll('.review-item')).find(el=>el.querySelector('span')?.textContent==='Сценарии');if(row){const value=row.querySelector('strong');value.textContent=[value.textContent,...enabled].filter(Boolean).join(' · ')||'Не выбраны'}};
const baseTemplateChange=$('live-template').onchange;
$('live-template').onchange=()=>{baseTemplateChange();setScenarioValues('live-extra',S.cameras.find(c=>c.name===$('live-template').value)?.config||{})};
const baseLiveNew=$('live-new').onclick;
$('live-new').onclick=()=>{const before=L.current?.id;baseLiveNew();if(L.current?.id!==before||L.drafting){setScenarioValues('live-extra');renderPpeAssessments('live-ppe-assessments',[])}};
const baseLiveStatus=renderLiveStatus;
renderLiveStatus=function(){baseLiveStatus();renderPpeAssessments('live-ppe-assessments',L.current?.assessments);if(!L.current)return;const active=['starting','running','stopping'].includes(L.current.state);for(const el of document.querySelectorAll('#live .scenario-extension input, #live-zone-kind'))el.disabled=active;if(active)setScenarioValues('live-extra',L.current.config||S.jobs.find(j=>j.id===L.current.job_id)?.config||{})};

for(const [parent,id] of [['live-sessions','live-ppe-assessments'],['result-events','result-ppe-assessments']])$(parent).insertAdjacentHTML('beforebegin',`<section id="${id}" class="callout compact" hidden></section>`);
const ppeTitles={helmet:'Каска',vest:'Жилет',mask:'Защита лица',harness:'Привязь'};
const ppeStates={worn:'Обнаружено',missing:'Отсутствует',unknown:'Не определено'};
const ppeReasons={pose_assisted_torso:'Корпус найден по измеренным плечам и бёдрам; результат классификатора требует проверки',pose_assisted_head:'Голова найдена по измеренным точкам лица; результат классификатора требует проверки',specialist_consensus:'Согласованы детектор и классификатор СИЗ',insufficient_pixels:'Слишком мало исходных пикселей для проверки головы',measured:'Свежее наблюдение',stronger_evidence:'Выбран более уверенный признак',conflicting_models:'Признаки противоречат друг другу',no_reliable_evidence:'Нет надёжного признака: проверьте видимость и размер области',recent_observation:'Последний результат; новое подтверждение не получено'};
function renderPpeAssessments(id,rows=[]){const el=$(id);el.hidden=!rows?.length;if(el.hidden)return;el.innerHTML='<b>Люди и СИЗ · последний обработанный кадр</b><small>Синяя рамка — слабое наблюдение или прогноз положения между замерами. Она не подтверждает новую тревогу.</small>'+rows.map(row=>`<div><strong>Человек #${Number(row.track_id)}</strong> · ${row.tracking==='predicted'?'Краткий пропуск · рамка сохранена':row.tracking==='weak'?'Слабое наблюдение':'Измеренная рамка'}${row.visibility?.small_objects==='limited'?'<p>Мелкие объекты: недостаточно деталей для надёжной проверки курения; нужна более близкая камера.</p>':''}${Object.entries(row.ppe||{}).map(([key,value])=>`<p><b>${esc(ppeTitles[key]||key)}: ${esc(ppeStates[value.state]||value.state)}</b> · ${esc(ppeReasons[value.reason]||value.reason)}${value.observed_at_s!=null?' · '+timestamp(value.observed_at_s):''}</p>`).join('')}</div>`).join('')}
const basePpeResult=renderResult;
renderResult=function(){basePpeResult();renderPpeAssessments('result-ppe-assessments',S.current?.stats?.latest_assessments)};
const baseSetStep=setStep;
setStep=function(step){baseSetStep(step);if(step===2)$('drawing-help').textContent=$('extra-harness')?.checked?'Выбран сценарий «Работа на высоте без привязи»: отметьте область с назначением «Работа на высоте», чтобы правило действовало только там. Высоту по видео система не измеряет. Если этот контроль не нужен, отключите сценарий на шаге «Сценарии».':'Если выбранный сценарий требует контекста, отметьте область и её назначение. Для каски и огня зоны необязательны. Разрешённые области исключают тревоги огня или курения.'};
const baseHelp=renderHelp;
renderHelp=function(){baseHelp();$('help-content').insertAdjacentHTML('afterbegin',`<article class="panel padded scenario-help"><h2>Новые сценарии: что настроить</h2><p><b>Каска и СИЗ:</b> выберите нужный контроль. Красная рамка означает явный отрицательный класс; отсутствие видимой экипировки само по себе тревогу не вызывает.</p><p><b>Контекст камеры:</b> разметьте область за ограждением, под грузом/конвейером, работы на высоте или оборудования. Для ограждения требуется наблюдаемый переход снаружи внутрь. Для привязи обязательна зона высоты.</p><p><b>Телефон и техника:</b> телефон сопоставляется с рукой и движением либо рабочей зоной. Дистанция до движущейся техники оценивается по изображению; для метров нужна калибровка пола.</p><p><b>Огонь и курение:</b> отметьте разрешённые области при их наличии. Каждый кандидат проходит выдержку и попадает в общий журнал с доказательством и клипом.</p><p><b>Граница возможностей:</b> переноска пока поддерживает только сильный наклон с распознанной сумкой/чемоданом. Масса груза, произвольные коробки, реальная высота и крепление страховки не определяются. Эти пункты требуют целевых данных и дообучения.</p></article>`)};
const baseQuality=renderQuality;
renderQuality=function(){baseQuality();$('quality-content').insertAdjacentHTML('beforeend',`<article class="panel padded"><h2>Новые сценарии: метрики пока не измерены</h2><p>Есть модели, временные правила и интеграционные проверки. Значения Precision, Recall и F1 для касок, огня, телефона и остальных новых сценариев ещё не получены на независимых размеченных видео. Результат URFD относится только к падениям.</p><p>Нужны целевые записи с отрицательными примерами: скрытая голова, перекрытия людей, пар, пыль, сварка, телефон в руке без использования и безопасная работа рядом с техникой.</p>${(S.quality.requirements?.case_scenarios||[]).map(r=>`<div class="readiness-row"><span class="pill ${r.status==='partial'?'amber':''}">${r.status==='partial'?'Частично':r.status==='measured_urfd'?'URFD: 0,90':'Нужна валидация'}</span><div><b>${esc(r.scenario)}</b><p>${esc(r.limitation)}</p></div></div>`).join('')}</article>`)};
renderHelp();
if(S.quality.status)renderQuality();

const baseParameterHelp={
 'zone-hold':'Сколько человек должен непрерывно находиться внутри зоны до тревоги. Например, 1 с отбрасывает краткое пересечение границы. Увеличение задерживает предупреждение.',
 'speed':'Порог относительной скорости. 1 рост/с — перемещение примерно на видимую высоту человека за секунду. Это не скорость в метрах; для м/с нужна калибровка пола.',
 'motion-hold':'Сколько должно сохраняться превышение порога скорости. Большая выдержка уменьшает случайные всплески трека, но увеличивает задержку.',
 'sample-fps':'Сколько кадров в секунду запрашивается для анализа записи. Это не гарантированная скорость нейросетей и не плавность LIVE. Больше кадров требует больше вычислений.',
 'body-threshold':'Порог оценки RGB-модели положения тела. 0,90 не означает точность 90%. Более высокий порог может уменьшить ложные тревоги и увеличить пропуски.',
 'body-hold':'Сколько должен сохраняться подтверждённый признак падения до тревоги. Уменьшение ускоряет реакцию, но повышает риск ложного события; длительное лежание проверяется отдельно.'
};
for(const [id,text] of Object.entries(baseParameterHelp)){
 const input=$(id);if(!input)continue;const helpId=id+'-explanation';
 input.setAttribute('aria-describedby',[input.getAttribute('aria-describedby'),helpId].filter(Boolean).join(' '));
 input.insertAdjacentHTML('afterend',`<small id="${helpId}" class="parameter-help">${text}</small>`);
}
