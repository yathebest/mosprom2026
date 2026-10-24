/* Interface localization only. User names, camera names, notes and input values
   are data, never translation keys. DOM nodes/handlers are preserved. */
(function(root){
  'use strict';
  const translations={
    'LinQ Safety · Контроль безопасности':'LinQ Safety · Safety monitoring',
    'Видеоаналитика безопасности':'Safety video analytics',
    'РАБОЧЕЕ ПРОСТРАНСТВО':'WORKSPACE','УПРАВЛЕНИЕ ПЛАТФОРМОЙ':'PLATFORM MANAGEMENT','ПЛАТФОРМА':'PLATFORM','ТЕКУЩЕЕ ПРЕДПРИЯТИЕ':'CURRENT ENTERPRISE',
    'Прямой эфир':'Live monitoring','Анализ записей':'Video analysis','Журнал событий':'Event log','Видеопроисшествия':'Incident clips','Очередь разбора':'Review queue','Статистика':'Statistics','Качество модели':'Model quality','Как это работает':'How it works',
    'Предприятия':'Enterprises','Сотрудники и роли':'Staff and roles','Журнал действий':'Audit log','Аккаунт':'Account','Безопасность на участке':'Workplace safety','Кабинет администратора':'Admin dashboard','← Кабинет администратора':'← Admin dashboard','Аккаунт и выход →':'Account and sign out →',
    'Данные на этом компьютере':'Data on this computer','Видео и доказательства хранятся локально.':'Videos and evidence are stored locally.','Проверяем оборудование…':'Checking hardware…','ПРОМЫШЛЕННАЯ БЕЗОПАСНОСТЬ':'INDUSTRIAL SAFETY','Как начать ↗':'Getting started ↗',
    '◉ Прямой эфир':'◉ Live monitoring','▣ Анализ записей':'▣ Video analysis','Камера · просмотр · алерты':'Camera · preview · alerts','Готовое видео · отчёт':'Recorded video · report',
    'НАБЛЮДЕНИЕ СЕЙЧАС':'LIVE MONITORING','Прямой эфир с детекцией':'Live monitoring with detection','Изображение с камеры, тревоги и автоматические клипы происшествий.':'Camera preview, alerts and automatic incident clips.',
    'Источник не подключён':'No source connected','Открыть происшествия →':'Open incidents →','Открыть видеопроисшествия →':'Open incident clips →','Здесь появится изображение с детекцией':'The camera preview will appear here','Подключите источник справа и нажмите «Начать эфир».':'Connect a source and select “Start live monitoring”.','Подключите источник и нажмите «Начать эфир».':'Connect a source and select “Start live monitoring”.','Инференс: —':'Inference: —',
    'Клип: 3 секунды до тревоги + 4 после.':'Clip: 3 seconds before the alert + 4 after.',
    'После накопления следующих кадров клип автоматически появится в «Видеопроисшествиях». Эфир продолжит работать.':'After the following frames arrive, the clip appears in Incident clips. Live monitoring continues.',
    'Подключение камеры':'Connect a camera','＋ Ещё камера':'＋ Another camera','1. Источник и кадр':'1. Source and frame','Источник':'Source','RTSP / HTTP-камера':'RTSP / HTTP camera','Веб-камера / USB-устройство':'Webcam / USB device','Адрес видеопотока':'Video stream URL','rtsp://… или http://…':'rtsp://… or http://…','URL с паролем хранится только в памяти до окончания эфира.':'The URL and its password remain in memory only until the session ends.','Устройство':'Device','Камера по умолчанию':'Default camera',
    'Предпросмотр включает камеру. Кнопка «Отключить камеру» освободит устройство без запуска эфира. Звук не записывается.':'Preview turns on the camera. Disconnect camera releases it without starting a session. Audio is not recorded.',
    'Получить кадр / проверить устройство':'Get a frame / check device','Отключить камеру':'Disconnect camera','Разметить зоны на кадре ↓':'Draw zones on the frame ↓','Камера выключена':'Camera disconnected','Название камеры':'Camera name','Правила сохранённой камеры':'Saved camera rules','Новые настройки':'New settings','2. Зоны на кадре':'2. Zones on the frame',
    'Получите кадр → отметьте минимум 3 вершины области → выберите назначение и сохраните зону.':'Get a frame → mark at least 3 polygon vertices → select a purpose and save the zone.',
    'Здесь размечаются зоны':'Draw zones here','Сначала получите кадр выбранной камеры. Во время разметки камеру можно отключить — кадр останется.':'Get a frame from the selected camera first. You can disconnect the camera while drawing; the frame remains.',
    'Получить кадр для разметки':'Get a frame for drawing','↶ Убрать точку':'↶ Undo point','Очистить точки':'Clear points','Название зоны':'Zone name','Сохранить зону':'Save zone','3. Сценарии и запуск':'3. Scenarios and start','Падение / человек на полу':'Fall / person down','Использовать зоны камеры':'Use camera zones','Быстрое перемещение':'Fast movement','Звуковой сигнал при новой тревоге':'Play a sound for new alerts','Начать эфир с детекцией ▶':'Start live monitoring ▶','Остановить эфир и сохранить клипы ■':'Stop monitoring and save clips ■',
    'До 2 эфиров одновременно. При остановке сохраняются и неполные последние клипы.':'Up to 2 simultaneous sessions. Stopping also saves the available part of the last clips.','События текущего эфира':'Events in this session','Все видеопроисшествия ↗':'All incident clips ↗','АРХИВ ВИДЕОДОКАЗАТЕЛЬСТВ':'VIDEO EVIDENCE ARCHIVE',
    'Каждый эпизод — отдельный клип с камерой, типом события и точным временем.':'Each episode has a separate clip with its camera, event type and timestamps.','Экспорт событий ↓':'Export events ↓','Все камеры':'All cameras','Все паттерны':'All patterns','Возможное падение':'Possible fall','Человек лежит':'Person down','Опасная зона':'Danger zone','Быстрое движение':'Fast movement','Любая готовность':'Any clip state','Клип готов':'Clip ready','Сохраняется':'Saving','Тайминги записи / эфира':'Video / session timestamps','От секунды':'From second','До секунды':'To second',
    'ОТ КАДРА К ДЕЙСТВИЮ':'FROM FRAME TO ACTION','Замечайте риски.':'Spot risks.','Проверяйте события.':'Review incidents.','Загрузите запись, выберите правила и получите видеофрагменты возможных нарушений.':'Upload a video, choose rules and get clips of possible violations.','Локальный анализ':'Local analysis','Добавить свою видеозапись':'Upload your video','Перетащите файл сюда или выберите на компьютере':'Drop a file here or choose one from your computer','MP4 · MOV · AVI · MKV · WebM до 500 МБ':'MP4 · MOV · AVI · MKV · WebM up to 500 MB','Выбрать видео ＋':'Choose video ＋','ГОТОВЫЙ ПРИМЕР':'READY-TO-USE EXAMPLE','Посмотрите весь путь':'Try the full workflow','на коротком видео':'on a short video',
    'Разберитесь с настройками на открытых примерах. Собственная камера не нужна.':'Explore the settings with public examples. No camera is needed.','Открытые учебные примеры · не финальный тест':'Public learning examples · not a final test','Есть адрес камеры?':'Have a camera URL?','Подключите RTSP или HTTP-поток. Эфир работает до остановки. URL не сохраняется на диск.':'Connect an RTSP or HTTP stream. Monitoring runs until stopped. The URL is not saved to disk.','Подключить камеру ↗':'Connect camera ↗','Выберите источник':'Choose a source','Своя запись или готовый пример':'Your video or a ready-made example','Настройте правила':'Configure rules','Сценарии, зоны, время подтверждения':'Scenarios, zones and confirmation time','Проверьте события':'Review events','Каждая тревога — с доказательством':'Evidence for every alert',
    'НОВЫЙ АНАЛИЗ':'NEW ANALYSIS','Другое видео':'Another video','Сценарии':'Scenarios','Зоны':'Zones','Проверка':'Review','ВИДЕО ГОТОВО':'VIDEO READY','Назовите источник':'Name the source','Начало записи (необязательно)':'Recording start (optional)','Нужно для статистики по времени суток. Если неизвестно, оставьте пустым.':'Used for time-of-day statistics. Leave empty if unknown.','Сохранённые настройки':'Saved settings','Какие ситуации проверять?':'Which scenarios should be checked?',
    'Включите нужные сценарии. Для опасной зоны потребуется разметка на следующем шаге.':'Enable the required scenarios. Danger zones are drawn in the next step.','Зоны и контекст камеры':'Zones and camera context','Опасные области, ограждения, груз, конвейер и контекст работы.':'Danger areas, barriers, suspended loads, conveyors and workplace context.','Полигон + трекинг':'Polygon + tracking','Падение и человек на полу':'Fall and person down','Быстрый переход к горизонтальному положению или длительное лежание.':'A rapid transition to a horizontal posture or prolonged lying down.','Проверяем наличие модели':'Checking model availability','Необычно высокая скорость движения по траектории человека.':'Unusually high movement speed along a person’s track.','Неподвижная камера · нужна настройка':'Fixed camera · calibration required','Система предлагает событие — решение принимает оператор.':'The system proposes an event; the operator makes the decision.','Откройте видеодоказательство и подтвердите или отклоните тревогу.':'Open the evidence and confirm or reject the alert.',
    'Время подтверждения и чувствительность':'Confirmation time and sensitivity','Время в опасной зоне, с':'Time inside a danger zone, s','Порог движения, ростов/с':'Movement threshold, body heights/s','Подтверждение движения, с':'Movement confirmation, s','Частота анализа, кадр/с':'Analysis sampling rate, frames/s','Порог положения тела':'Body posture threshold','Подтверждение падения, с':'Fall confirmation, s','Метрики падений относятся к порогу 0,90 и выдержке 0,5 с. При изменении настроек результат может отличаться. Скорость в ростах/с не является скоростью в метрах.':'Reported fall metrics use a 0.90 threshold and 0.5 s confirmation. Changing settings can change the result. Body heights/s is not metres/s.',
    'Где действует правило?':'Where does the rule apply?','Кликайте по границе области по порядку. Нужны минимум 3 точки, затем «Сохранить зону».':'Click the polygon boundary in order. Add at least 3 points, then choose Save zone.','ОПАСНАЯ ЗОНА · 0 ТОЧЕК':'DANGER ZONE · 0 POINTS','Начать заново':'Start again','Тестовая зона':'Example zone','Приоритет зоны':'Zone priority','Обычный':'Normal','Высокий':'High','Низкий':'Low','Проверяем положение ног':'The rule checks foot position','Используется нижний центр рамки. Пересечение верхней части рамки с зоной само по себе тревогу не вызывает.':'The lower centre of the bounding box is used. Overlap of the top of the box with a zone does not trigger an alert.',
    'Калибровка пола для скорости в м/с':'Floor calibration for speed in m/s','Использовать реальные размеры пола':'Use real floor dimensions','Отметьте 4 угла известного прямоугольника по порядку обхода. Эта разметка не заменяет опасную зону.':'Mark the 4 corners of a measured rectangle in order. This calibration does not replace a danger zone.','Ширина, м':'Width, m','Глубина, м':'Depth, m','Порог скорости, м/с':'Speed threshold, m/s','Отметить 4 угла пола':'Mark 4 floor corners','Калибровка не задана':'No calibration defined','Проверьте и запустите':'Review and start','События появятся во время анализа. Видеоклипы сохранятся после обработки кадров.':'Events appear during analysis. Clips are saved after the frames are processed.','Сохранить правила для этой камеры':'Save rules for this camera','Один продолжающийся эпизод — одно событие.':'One continuing episode produces one event.','Краткие срабатывания отбрасываются. После прекращения нарушения может быть зафиксирован новый эпизод.':'Brief detections are filtered out. A new episode may be recorded after the previous violation ends.','← Назад':'← Back','Продолжить →':'Continue →','Начать анализ ▶':'Start analysis ▶',
    '● Зелёный — человек ● Красный — правило активно':'● Green: person ● Red: active rule','1 · Подготовка':'1 · Setup','2 · Кадры':'2 · Frames','3 · Видеоклипы':'3 · Clips','4 · Готово':'4 · Complete','События':'Events','Скорость обработки':'Processing rate','кадров/с · этот компьютер':'frames/s · this computer','Модель · P95':'Model · P95','мс на обработанный кадр':'ms per analysed frame','Повторить с другими настройками':'Run again with other settings','Остановить анализ':'Stop analysis','События записи':'Events in this video','Весь журнал ↗':'Full event log ↗','ИСТОРИЯ РАБОТЫ':'ANALYSIS HISTORY','Ваши записи':'Your videos','＋ Добавить':'＋ Add',
    'ОТ ТРЕВОГИ К РЕШЕНИЮ':'FROM ALERT TO DECISION','Открывайте доказательства и отмечайте результат проверки.':'Open evidence and record your review.','Экспорт CSV ↓':'Export CSV ↓','Все события':'All events','Любой статус':'Any status','Нужна проверка':'Needs review','Подтверждено':'Confirmed','Ложное':'False alarm','Передано мастеру':'Forwarded to supervisor','Камеры, зоны и время':'Cameras, zones and time','Камера':'Camera','Зона':'Zone','Все зоны':'All zones','Дата обработки':'Processing date','От секунды видео':'From video second','До секунды видео':'To video second','СНАЧАЛА ВАЖНОЕ':'PRIORITY FIRST',
    'Тревоги по приоритету и времени ожидания. Цель проверки: 1 минута для высокого приоритета и 5 минут для остальных.':'Alerts ordered by priority and waiting time. Review target: 1 minute for high priority and 5 minutes for other alerts.','Разметка ZIP ↓':'Export labels ZIP ↓','Разметка улучшает следующий цикл обучения.':'Reviewed labels inform the next training cycle.','Подтверждённые и отклонённые события можно выгрузить с доказательствами. Автоматическое дообучение не запускается.':'Confirmed and rejected events can be exported with evidence. Retraining is not started automatically.','ОБЗОР НАБЛЮДЕНИЙ':'OBSERVATION SUMMARY','Где возникают события':'Where events occur','Статистика сохранённых прогонов и результатов проверки оператором.':'Statistics from saved runs and operator reviews.','ПРОВЕРЯЕМЫЕ РЕЗУЛЬТАТЫ':'MEASURED RESULTS','Качество, а не уверенность':'Quality beyond confidence','Вероятность на отдельном кадре не равна качеству на размеченных видео.':'A score on one frame is not accuracy on labelled videos.','Отчёт JSON ↗':'JSON report ↗','ПОНЯТНЫЙ ПРОЦЕСС':'A CLEAR WORKFLOW','Сначала задайте контекст камеры, затем проверяйте предложенные события.':'Configure the camera context first, then review proposed events.','Начать →':'Get started →','LinQ Safety · исследовательский прототип':'LinQ Safety · research prototype','Оператор проверяет каждое событие':'An operator reviews every event','РАЗБОР СОБЫТИЯ':'EVENT REVIEW',
    'Вход в опасную зону':'Entry into a danger zone','Человек длительно лежит':'Person down for a prolonged period','Без каски':'No hard hat','Без сигнального жилета':'No high-visibility vest','Без защиты лица':'No face protection','Без привязи в зоне высоты':'No harness in a height zone','Опасное сближение с техникой':'Unsafe proximity to machinery','Возможное использование телефона':'Possible phone use','Открытый огонь':'Open flame','Дым':'Possible smoke','Возможное курение':'Possible smoking','Вход за ограждение':'Barrier crossing','Человек под грузом':'Person under a suspended load','Человек под конвейером':'Person under a conveyor','Наклон при переноске · эксперимент':'Leaning while carrying · experimental',
    'Администратор платформы':'Platform administrator','Администратор предприятия':'Enterprise administrator','Специалист по охране труда':'Occupational safety specialist','Мастер / начальник смены':'Supervisor / shift manager','Оператор / диспетчер':'Operator / dispatcher','Работник / водитель':'Worker / driver','Служба охраны труда / юрист':'Safety service / legal reviewer','Менеджмент':'Management','Роль':'Role','Управление платформой':'Platform management',
    'КАБИНЕТ АДМИНИСТРАТОРА':'ADMIN DASHBOARD','КОМАНДА ПРЕДПРИЯТИЯ':'ENTERPRISE TEAM','ИСТОРИЯ ДЕЙСТВИЙ':'AUDIT HISTORY','ЛИЧНЫЙ АККАУНТ':'PERSONAL ACCOUNT','Заявки':'Applications','Активные':'Active','Всего':'Total','ожидают решения':'awaiting a decision','предприятия':'enterprises','рабочие пространства':'workspaces','На рассмотрении':'Pending approval','Подключено':'Active','Отклонено':'Rejected','Приостановлено':'Suspended','Без предприятия':'No enterprise','Сотрудник':'Staff member','Сотрудники':'Staff','Действие':'Action','Подробности':'Details','Время UTC':'Time (UTC)','Система':'System',
    'Вход':'Sign in','Пароль изменён':'Password changed','Создан администратор платформы':'Platform administrator created','Открыто пространство предприятия':'Enterprise workspace opened','Заявка предприятия':'Enterprise application','Предприятие подключено':'Enterprise activated','Заявка отклонена':'Application rejected','Доступ приостановлен':'Access suspended','Повторная заявка':'Application resubmitted','Сотрудник добавлен':'Staff member added','Роль / доступ изменены':'Role / access changed','Пароль сброшен':'Password reset','Событие проверено':'Event reviewed',
    'Принять заявку':'Approve application','Отклонить':'Reject','Открыть пространство →':'Open workspace →','Приостановить доступ':'Suspend access','Возобновить доступ':'Restore access','Отправить повторно':'Resubmit','Выйти из аккаунта':'Sign out','Задайте постоянный пароль':'Set a permanent password','Изменить пароль':'Change password','После изменения все остальные сессии будут завершены.':'Changing your password signs out all other sessions.','Текущий пароль':'Current password','Новый пароль, от 12 символов':'New password, at least 12 characters','Повтор нового пароля':'Repeat new password','Сохранить пароль':'Save password','Новые пароли не совпадают':'New passwords do not match',
    'Открываем рабочее пространство…':'Opening the workspace…','Проверяем права доступа и загружаем данные предприятия.':'Checking permissions and loading enterprise data.','Логин':'Username','Пароль':'Password','Имя':'Name','Имя сотрудника':'Staff member name','Войти':'Sign in','Войдите в аккаунт':'Please sign in','Подать заявку':'Apply','Заявка на подключение предприятия':'Apply to connect an enterprise','Название предприятия':'Enterprise name','Контактное описание':'Contact details','Отправить заявку':'Submit application','Сохранить':'Save','Отмена':'Cancel','Удалить':'Remove','Добавить сотрудника':'Add staff member','Отключить':'Disable','Восстановить':'Restore','Все камеры предприятия':'All enterprise cameras','Назначенные камеры':'Assigned cameras','Выберите роль':'Choose a role','Активен':'Active','Отключён':'Disabled',
    'Журнал и тревоги':'Event log and alerts','Просмотр эфиров':'View live sessions','Записи камер':'Camera recordings','Агрегированная статистика':'Aggregate statistics','Кадры и клипы':'Frames and clips','Экспорт доказательств':'Export evidence','Правила и зоны':'Rules and zones','Анализ видео':'Video analysis','Подключение и остановка камер':'Connect and stop cameras','Разбор событий':'Review events','История действий':'Audit history',
    'Доказательство ↗':'View evidence ↗','К моменту тревоги →':'Go to alert time →','Кадр с разметкой':'Annotated frame','Зона / область':'Zone / area','Весь кадр':'Entire frame','Начало → тревога в видео':'Onset → alert in video','Почему возникло событие':'Why this event was raised','Результат проверки':'Review result','Фрагмент исходной записи, без наложенной разметки.':'Clip from the original recording, without annotations.','Анализ завершён':'Analysis complete','Готово к анализу':'Ready for analysis','В очереди':'Queued','Выполняется':'Running','Сохраняем видеоклипы':'Saving clips','Ошибка':'Error','Отменено':'Cancelled','Эфир остановлен':'Live session stopped','Эфир запускается':'Starting live session','Прямой эфир работает':'Live monitoring is running','Останавливаем эфир':'Stopping live monitoring',
    'Готовность клипа':'Clip state','Камера происшествия':'Incident camera','Паттерн происшествия':'Incident pattern','Дата происшествия':'Incident date','Тип события':'Event type','Статус события':'Event status','Поиск событий':'Search events','Поиск по камере, зоне, типу':'Search camera, zone or event type','Закрыть событие':'Close event','Скрыть алерт':'Dismiss alert','Редактор зон':'Zone editor','Зоны прямого эфира':'Live zone editor','Режим работы':'Operating mode','Прямой эфир с результатами детекции':'Live monitoring with detection results',
    'За ограждением':'Behind a barrier','Под подвешенным грузом':'Under a suspended load','Под конвейером':'Under a conveyor','Работа на высоте':'Work at height','Работающее оборудование':'Operating equipment','Курение разрешено':'Smoking permitted','Огонь разрешён':'Hot work permitted','Зеркало · исключить отражение по положению ног':'Mirror · ignore reflected foot positions','Назначение зоны':'Zone purpose',
    'Люди и СИЗ · последний обработанный кадр':'People and PPE · last analysed frame','Каска':'Hard hat','Жилет':'Vest','Защита лица':'Face protection','Привязь':'Harness','Обнаружено':'Detected','Отсутствует':'Missing','Не определено':'Unknown','Измеренная рамка':'Measured bounding box','Слабое наблюдение':'Weak observation','Краткий пропуск · рамка сохранена':'Short gap · last box retained','Согласованы детектор и классификатор СИЗ':'PPE detector and classifier agree','Слишком мало исходных пикселей для проверки головы':'Too few native pixels to assess the head','Свежее наблюдение':'Fresh observation','Выбран более уверенный признак':'Stronger evidence selected','Признаки противоречат друг другу':'Conflicting evidence','Нет надёжного признака: проверьте видимость и размер области':'No reliable evidence: check visibility and region size','Последний результат; новое подтверждение не получено':'Previous result; no fresh confirmation',
    'Синяя рамка — слабое наблюдение или прогноз положения между замерами. Она не подтверждает новую тревогу.':'A blue box is a weak observation or a projected position between measurements. It cannot confirm a new alert.',
    'Корпус найден по измеренным плечам и бёдрам; результат классификатора требует проверки':'Torso proposed from measured shoulders and hips; review the classifier result','Голова найдена по измеренным точкам лица; результат классификатора требует проверки':'Head proposed from measured facial landmarks; review the classifier result','Мелкие объекты: недостаточно деталей для надёжной проверки курения; нужна более близкая камера.':'Small objects: too little detail to assess smoking reliably; a closer camera is needed.',
    'Язык интерфейса':'Interface language','Переключить язык интерфейса':'Choose interface language'
  };
  const canonical=new WeakMap(),attributeValues=new WeakMap();
  let selected='ru';try{selected=root.localStorage?.getItem('linq.language')==='en'?'en':'ru'}catch(_){}
  const normalize=s=>String(s).replace(/\s+/g,' ').trim();
  const dynamic=[
    [/^(\d+) обработанных кадров$/,m=>`${m[1]} analysed frames`],[/^(\d+) событий$/,m=>`${m[1]} events`],
    [/^(\d+) \/ 8 зон$/,m=>`${m[1]} / 8 zones`],[/^(\d+) точек · нужно не менее 3$/,m=>`${m[1]} points · at least 3 required`],
    [/^Человек #(\d+)$/,m=>`Person #${m[1]}`],[/^Шаг (\d+) из (\d+)$/,m=>`Step ${m[1]} of ${m[2]}`],
    [/^(\d+) в работе · (\d+) в очереди · до (\d+) параллельно$/,m=>`${m[1]} running · ${m[2]} queued · up to ${m[3]} in parallel`],
    [/^(\d+) ожидают проверки$/,m=>`${m[1]} awaiting review`],[/^(\d+) событий · (\d+) клипов в работе$/,m=>`${m[1]} events · ${m[2]} clips in progress`],
    [/^Видео ([\d.]+|—) кадр\/с · люди ([\d.]+|—) · СИЗ\/объекты ([\d.]+|—)$/,m=>`Video ${m[1]} frames/s · people ${m[2]} · PPE/objects ${m[3]}`],
    [/^Анализ людей P95: ([\d.]+|—) мс · задержка СИЗ ([\d.]+|—) мс$/,m=>`People analysis P95: ${m[1]} ms · PPE result age ${m[2]} ms`],
    [/^Точка (\d+) · удалить$/,m=>`Point ${m[1]} · remove`],[/^Удалить точку (\d+)$/,m=>`Remove point ${m[1]}`],
    [/^(\d+) точек$/,m=>`${m[1]} points`],[/^отключено: (\d+)$/,m=>`disabled: ${m[1]}`],
    [/^Сотрудники · (\d+)$/,m=>`Staff · ${m[1]}`],[/^Сотрудников: (\d+) активных \/ (\d+) всего$/,m=>`Staff: ${m[1]} active / ${m[2]} total`],
    [/^(\d+)% · (.+)$/,m=>`${m[1]}% · ${translate(m[2],'en')}`],
    [/^Видео ([\d.:]+) → тревога ([\d.:]+)$/,m=>`Video ${m[1]} → alert ${m[2]}`],
    [/^Итоговая проверка: (\d+) коротких роликов URFD\. Это результат для сценария падения, а не для всех нарушений и камер\.$/,m=>`Final evaluation: ${m[1]} short URFD videos. This result concerns falls, not all violations and cameras.`],
    [/^Проверено (\d+) с обычных действий\. Пересчёт на час: ([\d.]+) ложных тревог\/ч; при таком коротком наборе оценка нестабильна\.$/,m=>`${m[1]} s of ordinary activities evaluated. Extrapolated rate: ${m[2]} false alarms/hour; this short sample gives an unstable estimate.`],
    [/^Модель: RGB ResNet18 \+ временные правила$/,()=>`Model: RGB ResNet18 + temporal rules`],
    [/^порог ([\d.]+)$/,m=>`threshold ${m[1]}`],[/^выдержка ([\d.]+) с$/,m=>`confirmation ${m[1]} s`],
    [/^Начало события ([\d.:]+)$/,m=>`Event onset ${m[1]}`],[/^тревога ([\d.:]+)$/,m=>`alert ${m[1]}`],
    [/^Люди: (\d+)$/,m=>`People: ${m[1]}`],[/^Всего: (\d+)$/,m=>`Total: ${m[1]}`]
  ];
  function translate(value,language=selected,protectedValues=null){
    if(language!=='en')return String(value);
    const input=String(value),key=normalize(input);if(protectedValues?.has(key))return input;let result=translations[key];
    if(result===undefined){for(const [pattern,format] of dynamic){const match=key.match(pattern);if(match){result=format(match);break}}}
    if(result===undefined&&key.includes(' · ')){
      const parts=key.split(' · '),changed=parts.map(x=>translate(x,'en',protectedValues));
      if(changed.some((x,i)=>x!==parts[i]))result=changed.join(' · ');
    }
    if(result===undefined){const icon=key.match(/^([◉◫☷▣⚑▥◎♙≡○◇?]+)\s*(.+)$/);if(icon&&translations[icon[2]])result=icon[1]+' '+translations[icon[2]]}
    return result===undefined?input:input.replace(input.trim(),result);
  }
  function userValues(){
    const values=new Set();const add=value=>{if(typeof value==='string'&&value.trim())values.add(normalize(value))};
    try{
      if(typeof S!=='undefined'){
        for(const c of S.cameras||[])add(c.name);
        for(const j of S.jobs||[]){add(j.filename);add(j.config?.camera);for(const z of j.config?.zones||[])add(z.name)}
        for(const e of S.events||[]){add(e.camera);add(e.zone)}
        for(const z of S.zones||[])add(z.name);
      }
      if(typeof A!=='undefined'){
        add(A.user?.name);add(A.user?.login);add(A.user?.organization?.name);
        for(const o of A.organizations||[]){add(o.name);add(o.note);add(o.reason)}
        for(const m of A.members||[]){add(m.name);add(m.login)}
        for(const c of A.teamCameras||[])add(c.name);
      }
      if(typeof L!=='undefined'){add(L.current?.camera);for(const s of L.sessions||[])add(s.camera);for(const z of L.zones||[])add(z.name)}
    }catch(_){}
    return values;
  }
  function ignored(element){return !element||!element.closest||Boolean(element.closest('[data-i18n-ignore],script,style,code,pre,textarea,input,[contenteditable="true"]'))}
  function preserveUserText(element,text,users){
    if(element.closest('nav,[data-page],.language-control'))return false;
    if(element.matches('option')&&element.value&&element.parentElement?.matches('#live-template,#template,#filter-camera,#filter-zone,#clip-camera'))return true;
    return users.has(normalize(text));
  }
  function apply(scope){
    if(typeof document==='undefined')return;
    scope=scope||document.body;if(!scope)return;const users=userValues();
    const walker=document.createTreeWalker(scope,NodeFilter.SHOW_TEXT);const nodes=[];
    while(walker.nextNode())nodes.push(walker.currentNode);
    for(const node of nodes){
      const element=node.parentElement;if(ignored(element))continue;
      const current=node.data,prior=canonical.get(node);
      const source=prior&&current===prior.rendered?prior.source:current;
      if(preserveUserText(element,source,users))continue;
      const rendered=translate(source,selected,element.closest('nav,[data-page],.language-control')?null:users);canonical.set(node,{source,rendered});if(current!==rendered)node.data=rendered;
    }
    for(const element of [scope,...scope.querySelectorAll('[placeholder],[title],[aria-label]')]){
      if(ignored(element)||!element.getAttribute)continue;
      const saved=attributeValues.get(element)||{};
      for(const key of ['placeholder','title','aria-label']){
        const current=element.getAttribute(key);if(current===null)continue;
        const prior=saved[key],source=prior&&prior.rendered===current?prior.source:current;
        const rendered=translate(source);saved[key]={source,rendered};if(current!==rendered)element.setAttribute(key,rendered);
      }
      attributeValues.set(element,saved);
    }
  }
  function setLanguage(language){
    selected=language==='en'?'en':'ru';try{root.localStorage?.setItem('linq.language',selected)}catch(_){}
    if(typeof document!=='undefined'){
      document.documentElement.lang=selected;for(const id of ['interface-language','auth-interface-language']){const select=document.getElementById(id);if(select)select.value=selected}
      apply();document.title=translate('LinQ Safety · Контроль безопасности');
      document.dispatchEvent(new CustomEvent('linq:languagechange',{detail:{language:selected,locale:locale()}}));
    }
    return selected;
  }
  function locale(){return selected==='en'?'en-GB':'ru-RU'}
  const api={t:translate,language:()=>selected,locale,apply,setLanguage,translations};root.LinQI18n=api;
  if(typeof module!=='undefined'&&module.exports)module.exports=api;
  if(typeof document!=='undefined'){
    document.documentElement.lang=selected;
    const start=()=>{
      for(const id of ['interface-language','auth-interface-language']){const select=document.getElementById(id);if(select){select.value=selected;select.addEventListener('change',()=>setLanguage(select.value))}}
      setLanguage(selected);let queued=false;
      new MutationObserver(()=>{if(queued)return;queued=true;requestAnimationFrame(()=>{queued=false;apply()})}).observe(document.body,{childList:true,subtree:true,characterData:true,attributes:true,attributeFilter:['placeholder','title','aria-label']});
    };
    if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',start,{once:true});else start();
  }
})(typeof globalThis!=='undefined'?globalThis:this);
