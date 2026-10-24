"""Create a 16:9 PDF presentation of the complete LinQ Safety prototype."""
from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph,Table,TableStyle
from reportlab.lib.utils import ImageReader

ROOT=Path(__file__).resolve().parents[1]
W,H=1280,720
DARK=colors.HexColor('#153E40');GREEN=colors.HexColor('#C5E581')
PALE=colors.HexColor('#F3F6F0');TEXT=colors.HexColor('#183E43');MUTED=colors.HexColor('#52686B')
LINE=colors.HexColor('#D5E0D5');AMBER=colors.HexColor('#9A5B16')


def main():
    for name,file in [('Deck','segoeui.ttf'),('DeckBold','segoeuib.ttf')]:pdfmetrics.registerFont(TTFont(name,'C:/Windows/Fonts/'+file))
    pdfmetrics.registerFontFamily('Deck',normal='Deck',bold='DeckBold',italic='Deck',boldItalic='DeckBold')
    target=ROOT/'output/pdf/LinQ-Safety-project-presentation-2026-10-24.pdf';target.parent.mkdir(parents=True,exist_ok=True)
    c=canvas.Canvas(str(target),pagesize=(W,H));c.setTitle('LinQ Safety - весь проект');c.setAuthor('Команда LinQ Safety')
    overflow=[];page=0

    def text(s,x,y,width=1152,size=24,color=TEXT,bold=False,leading=None):
        s=s.replace('\u2011','-').replace('\u2013','-').replace('\u2014','-')
        style=ParagraphStyle('s',fontName='DeckBold' if bold else 'Deck',fontSize=size,leading=leading or size*1.36,textColor=color)
        p=Paragraph(s,style);_,height=p.wrap(width,H)
        p.drawOn(c,x,y-height)
        if y-height<67:overflow.append((page,s[:70],y-height))
        return y-height

    def begin(title,subtitle=None,dark=False):
        nonlocal page
        page+=1;c.setFillColor(DARK if dark else PALE);c.rect(0,0,W,H,stroke=0,fill=1)
        c.setFillColor(GREEN if dark else DARK);c.setFont('DeckBold',15);c.drawString(64,677,'LinQ Safety')
        c.setFont('Deck',14);c.drawRightString(W-64,677,'24.10.2026')
        c.setStrokeColor(colors.HexColor('#456264') if dark else LINE);c.line(64,655,W-64,655)
        text(title,64,628,size=39,color=colors.white if dark else TEXT,bold=True)
        if subtitle:text(subtitle,64,566,size=20,color=GREEN if dark else MUTED)
        c.setFillColor(colors.HexColor('#B9CDCB') if dark else MUTED);c.setFont('Deck',13)
        c.drawString(64,30,'Демонстрационный прототип. Статусы и границы проверки указаны в материале.')
        c.drawRightString(W-64,30,f'{page:02d} / 16')

    def bullets(items,x=64,y=520,width=540,size=25,gap=22,color=TEXT):
        for item in items:
            c.setFillColor(GREEN);c.circle(x+5,y-14,5,stroke=0,fill=1)
            y=text(item,x+25,y,width-25,size=size,color=color)-gap
        return y

    def image(path,x,y,width,height):
        # Preserve the entire screenshot/frame: no evidence is cropped away.
        im=ImageReader(str(path));iw,ih=im.getSize();scale=min(width/iw,height/ih)
        rw,rh=iw*scale,ih*scale;c.drawImage(im,x+(width-rw)/2,y+(height-rh)/2,width=rw,height=rh,mask='auto')

    def rule(x,y,width):c.setStrokeColor(LINE);c.line(x,y,x+width,y)

    def table(rows,x=64,top=530,widths=(300,850),size=20):
        styles=[];cells=[]
        for i,row in enumerate(rows):
            style=ParagraphStyle('cell',fontName='DeckBold' if i==0 else 'Deck',fontSize=size,leading=size*1.28,
                                 textColor=colors.white if i==0 else TEXT)
            cells.append([Paragraph(value,style) for value in row])
        t=Table(cells,colWidths=list(widths),repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),DARK),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#E8F0E4')]),
            ('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),11),('BOTTOMPADDING',(0,0),(-1,-1),11),
            ('LEFTPADDING',(0,0),(-1,-1),14),('RIGHTPADDING',(0,0),(-1,-1),14)]))
        _,h=t.wrap(W,H);t.drawOn(c,x,top-h)
        if top-h<75:overflow.append((page,'table',top-h))
        return top-h

    # 1. Product framing.
    begin('Видеоаналитика промышленной безопасности',dark=True)
    text('От камеры<br/>к проверенному<br/>происшествию',64,535,650,54,colors.white,True,65)
    text('Люди, правила участка, тревоги и видеодоказательства в одном рабочем пространстве.',68,300,610,26,GREEN)
    image(ROOT/'docs/assets/factory-context-20261024.jpg',735,250,480,300)
    text('Открытая заводская запись из набора Önal / Dandıl.<br/>Используется для практической проверки.',750,220,450,17,colors.HexColor('#B9CDCB'))
    c.showPage()

    # 2. End-to-end workflow.
    begin('Система выделяет моменты, требующие внимания','Камеры дают непрерывное видео. Оператору нужен конкретный эпизод с объяснением и доказательством.')
    stages=[('01','Источник','Камера или файл'),('02','Распознавание','Люди, поза, объекты'),('03','Правила','Зона + время + контекст'),('04','Происшествие','Алерт, кадр и клип'),('05','Решение','Проверка оператором')]
    for i,(n,title,body) in enumerate(stages):
        x=64+i*234;c.setFillColor(colors.white);c.rect(x,355,214,150,stroke=0,fill=1)
        text(n,x+14,490,185,30,DARK,True);text(title,x+14,445,185,22,TEXT,True);text(body,x+14,408,185,18,MUTED)
        if i<4:
            c.setStrokeColor(DARK);c.setLineWidth(2);c.line(x+217,430,x+231,430);c.line(x+226,435,x+231,430);c.line(x+226,425,x+231,430)
    text('Что получает предприятие',64,295,size=27,bold=True)
    bullets(['Поиск событий вместо ручного просмотра всей записи.','Единый журнал и понятная ответственность за разбор.','Сохранённый контекст: где, когда и почему возникла тревога.'],64,248,1120,23,12)
    c.showPage()

    # 3. Two first-class modes.
    begin('Два режима для одной системы правил')
    for x,title,items in [(64,'Прямой эфир',['RTSP / HTTP-источник или веб-камера / USB.','Просмотр текущего видео и тревоги во время работы.','Автоматическая запись фрагмента происшествия.']),
                          (672,'Анализ записей',['Загрузка видео через пошаговую форму.','Выбор сценариев, зон и времени начала записи.','Размеченное видео, клипы, события и отчёт.'])]:
        rule(x,537,540);text(title,x,515,540,31,bold=True);bullets(items,x,455,540,24,25)
    text('Общее: настройки камеры, роли доступа, журнал, статусы проверки и архив доказательств.',64,214,1150,25,DARK,True)
    text('Локальный прототип допускает два LIVE-сеанса и два фоновых анализа файлов. Масштабирование на множество предприятий требует отдельных нагрузочных испытаний.',64,130,1150,18,MUTED)
    c.showPage()

    # 4. Real interface.
    begin('LIVE: видео, события и последние оценки')
    image(ROOT/'docs/assets/project-live-20261024.png',665,105,550,465)
    bullets(['Источник и правила выбираются для конкретной камеры.','Тревога содержит паттерн, камеру и тайминги.','СИЗ отображаются с состоянием и причиной неопределённости.','FPS видео и частота нейроанализа показаны отдельно.'],64,530,560,24,23)
    text('Реальный скриншот тестового LIVE из зацикленной заводской записи. Число событий этого теста не является метрикой полноты или повторов.',64,155,545,17,MUTED)
    c.showPage()

    # 5. Evidence playback.
    begin('Запись превращается в проверяемый эпизод')
    image(ROOT/'docs/assets/project-evidence-20261024.png',64,92,590,475)
    bullets(['Исходный клип и кадр с разметкой.','Начало наблюдения и момент подтверждения тревоги.','Причина правила и статус проверки.','Переход к моменту события и скачивание MP4.'],710,527,500,24,24)
    text('LIVE-клип: 3 секунды до тревоги + 4 после. При недостатке кадров сохраняется доступная часть с соответствующей отметкой.',710,228,500,20,MUTED)
    text('Пример: «Без каски» на заводской записи; это отдельная функциональная проверка, не независимая оценка всех людей.',710,134,500,17,MUTED)
    c.showPage()

    # 6. Operator lifecycle.
    begin('Оператор управляет разбором, а не потоком рамок')
    table([['Раздел сайта','Работа пользователя'],
           ['Журнал событий','Фильтры по камере, паттерну, зоне, времени и статусу.'],
           ['Видеопроисшествия','Просмотр готовых клипов; состояния записи, экспорта и ошибок.'],
           ['Очередь разбора','Порядок по типу события, приоритету зоны и времени ожидания.'],
           ['Статистика','Распределение событий по типам, камерам и времени.'],
           ['Качество модели','Протоколы и ограничения измерений; уверенность кадра отделена от точности.']],widths=(300,850),size=21)
    text('Статусы: нужна проверка → подтверждено / ложное / передано мастеру.',64,174,1150,24,DARK,True)
    text('Результат проверки записывается с автором и временем. Экспорт доказательств и разметки готовит материал для следующего обучения.',64,128,1150,19,MUTED)
    c.showPage()

    # 7. Context rather than frame-only guesses.
    begin('Камера задаёт геометрию и контекст участка')
    bullets(['Зоны рисуются полигонами на исходном кадре. Проверяется положение ног человека.','Назначения: опасная область, ограждение, груз, конвейер, высота, оборудование.','Разрешённые области огня/курения и зона зеркала помогают задать правила участка.','Четыре точки пола и известные размеры позволяют оценивать скорость в м/с.'],64,535,1135,25,21)
    rule(64,235,1150)
    text('Важная граница',64,210,1100,26,bold=True)
    text('Полигон задаёт контекст оператора. Система не измеряет массу груза, реальную высоту конструкции или правильность крепления страховки по одной отметке зоны.',64,161,1135,25,MUTED)
    c.showPage()

    # 8. Multi-tenancy and roles.
    begin('Предприятия и роли изолируют рабочие данные')
    table([['Роль','Доступ в пределах своей области'],
           ['Администратор платформы','Заявки, предприятия, приглашения локальных администраторов; явный вход в предприятие.'],
           ['Администратор предприятия','Камеры, правила, сотрудники, роли и аудит своего предприятия.'],
           ['Специалист по охране труда','Правила, анализ, LIVE, проверка событий, экспорт и статистика.'],
           ['Мастер / оператор','Мастер: назначенные камеры участка. Оператор: эфиры, события и разбор; без изменения ролей.'],
           ['Работник / водитель','Памятка безопасности и собственный аккаунт.'],
           ['Юрист / менеджмент','Юрист: доказательства и история. Менеджмент: агрегированная статистика.']],top=560,widths=(330,820),size=18)
    text('Приглашение администратора: одноразовая ссылка на 60 секунд. Сейчас локальная имитация; SMTP-транспорт подготовлен к настройке.',64,158,1150,20,MUTED)
    text('Права проверяет сервер, включая прямые ссылки на медиа. Смена роли или отключение сотрудника отзывает его сессии.',64,93,1150,17,MUTED)
    c.showPage()

    # 9. Neural architecture.
    begin('Распознавание разделено на специализированные этапы')
    table([['Этап','Текущая реализация'],
           ['Люди и поза','COCO + дообученный YOLO12m; YOLOv8n-pose; дополнительные масштабы и участки исходного кадра.'],
           ['Трекинг','ByteTrack и консервативное кратковременное связывание ID.'],
           ['Каски / жилеты','YOLO12m-детекторы областей + ResNet50-классификаторы головы и торса.'],
           ['Падения','ResNet18 состояния сцены и ResNet18/GRU временного окна; заводская точность остаётся слабым местом.'],
           ['Другие объекты','ONNX-детекторы телефона, сигареты, огня/дыма и техники. Новые кандидаты проходят отдельную приёмку.']],widths=(260,890),size=21)
    text('FastAPI + HTML/CSS/JS · SQLite · OpenCV · PyTorch / ONNX Runtime · FFmpeg H264',64,103,1150,22,DARK,True)
    c.showPage()

    # 10. Case coverage.
    begin('Сценарии кейса объединены одним циклом событий')
    table([['Группа','Что предусмотрено','Состояние'],
           ['Поведение','Опасная зона, быстрое перемещение, падение / лежание.','Правила реализованы; качество зависит от сценария.'],
           ['СИЗ','Каска, жилет, защита лица; привязь в зоне высоты.','Есть модели и явная неопределённость.'],
           ['Объекты и инциденты','Телефон у руки, курение, огонь / дым, сближение с техникой.','Экспериментальные промышленные сценарии.'],
           ['Контекст участка','За ограждением, под грузом, под конвейером.','Зоны оператора, не распознавание самой конструкции.'],
           ['Дополнительная задача','Сильный наклон при переноске.','Частично: поддержанные предметы, без массы и произвольных коробок.']],top=550,widths=(210,535,405),size=20)
    text('Наличие функции в интерфейсе и подтверждённое качество распознавания - разные уровни готовности.',64,145,1150,23,AMBER,True)
    c.showPage()

    # 11. Training/data.
    begin('Данные расширяются под разные уровни задачи')
    table([['Набор','Как используется'],
           ['PPE-корпус','24 288 уникальных изображений после фильтрации; 108 416 crop головы/торса. Crop не считаются независимыми фотографиями.'],
           ['Заводские записи','64 клипа Safe / Unsafe Behaviours для осмотра и проверок. Поведенческие классы не подменяют разметку СИЗ.'],
           ['FPI-Det','22 766 пригодных кадров после очистки; телефон и голова. Source class mapping сохраняется явно.'],
           ['CigDet','557 изображений с боксами сигарет; дополнительные train-варианты с уменьшением разрешения.'],
           ['URFD и iSafetyBench','Лабораторные падения + рабочие сцены для выявления переноса домена и сложных отрицательных примеров.']],top=555,widths=(280,870),size=18.5)
    text('Выбор - по validation; test отложен. Дубликаты не переходят между частями.',64,143,1150,21,DARK,True)
    text('Происхождение и хеши - в манифестах. Сохраняются checkpoints и лучшие веса.',64,94,1150,17,MUTED)
    c.showPage()

    # 12. LinQ deployment.
    begin('LinQ: подготовлен путь от модели до ускорителя')
    names=['PyTorch','ONNX','SDK LinQ','.tpu','Устройство']
    for i,name in enumerate(names):
        x=64+i*234;c.setFillColor(GREEN if i<4 else colors.HexColor('#E9DBC4'));c.rect(x,385,210,105,stroke=0,fill=1)
        text(name,x+12,455,186,27,DARK,True)
        if i<4:
            c.setStrokeColor(DARK);c.setLineWidth(2);c.line(x+214,437,x+231,437);c.line(x+226,442,x+231,437);c.line(x+226,432,x+231,437)
    text('Проверено',64,333,540,28,bold=True)
    text('COCO YOLO12m скомпилирована SDK 2026.9.45 в TPU_H1M / O1. Калибровка на 8 train-кадрах проверяет тракт, а не качество всей модели.',64,286,550,24)
    text('Следующая проверка',674,333,540,28,bold=True)
    text('Новые веса, точность после квантования, фактический инференс и задержка на устройстве. Вход 960 требует отдельного статического варианта.',674,286,540,24)
    text('Успешная компиляция и работа на локальной RTX 4060 не подтверждают аппаратную производительность LinQ.',64,145,1150,23,AMBER,True)
    c.showPage()

    # 13. Honest evidence.
    begin('Практические результаты и границы измерений')
    table([['Проверка','Факт','Как интерпретировать'],
           ['Люди на 8 заводских кадрах','16/25 → 19/25','Грубые ручные боксы, IoU 0,30; отладочная выборка.'],
           ['63 выбранные открытые головы','41/63 → 56/63','Пропуски / unknown учитываются; не event F1 всей фабрики.'],
           ['LIVE-предпросмотр','17-20 FPS','Замер до последних изменений; нейроанализ около 1 FPS.'],
           ['Логика и доступ','117 Python + 9 JS','Тесты не заменяют оценку нейросетей.'],
           ['Падения на рабочих сценах','Выявлены пропуски и ложные тревоги','Лабораторный F1 не переносится автоматически.']],widths=(340,245,565),size=20)
    text('Цель Precision / Recall / F1 > 0,95 для всех заводских событий пока не достигнута.',64,115,1150,27,AMBER,True)
    c.showPage()

    # 14. Added product features.
    begin('Собственные функции усиливают базовую детекцию')
    bullets(['Клип с предысторией происшествия без остановки эфира.','Объяснение unknown: невидимая голова, маленький объект, конфликт моделей.','Очередь по приоритету зоны и времени ожидания.','Подтверждение оператором и экспорт разметки для следующего обучения.','Калибровка пола, карта присутствия и диагностика качества изображения.','Рабочие пространства предприятий, роли, приглашения и аудит доступа.'],64,535,1140,25,20)
    text('Отдельная текущая доработка: навигация, редактирование зон, читаемость аудита, пояснения настроек и RU/EN.',64,107,1150,21,MUTED)
    c.showPage()

    # 15. Demo talk track.
    begin('Как показать проект на защите','Последовательность демонстрации для команды и жюри')
    for i,(title,body) in enumerate([
        ('Настройка камеры','Выбрать источник, сценарий и область действия. Объяснить, что задаёт оператор.'),
        ('Наблюдение','Запустить LIVE, показать плавное видео, состояние модели и отдельные счётчики FPS.'),
        ('Происшествие','Показать алерт, тайминги, исходный клип и кадр с разметкой.'),
        ('Решение оператора','Подтвердить / отклонить событие, открыть журнал и статистику.'),
        ('Управление и развитие','Показать предприятие и роли; объяснить обучение, проверку кандидатов и перенос на LinQ.')]):
        y=520-i*83;text(f'{i+1:02d}',64,y,65,31,DARK,True);text(title,150,y,320,25,TEXT,True);text(body,505,y,690,23)
    c.showPage()

    # 16. Readiness, sources and next gates.
    begin('Что готово и что проверяем дальше',dark=True)
    text('Уже есть',64,535,530,30,GREEN,True)
    bullets(['Два режима сайта и общий цикл событий.','Роли, предприятия, журнал и доказательства.','Обученные модели и воспроизводимые проверки.'],64,475,540,25,24,color=colors.white)
    text('Перед полным закрытием кейса',675,535,540,30,GREEN,True)
    bullets(['Проверка новых кандидатов телефона / курения.','Улучшение падений и различения дыма / пыли.','Аппаратный запуск LinQ и измерение качества / задержки.'],675,475,540,25,24,color=colors.white)
    text('Материалы проекта',64,227,1100,25,GREEN,True)
    text('<a href="https://github.com/yathebest/mosprom2026" color="#C5E581">github.com/yathebest/mosprom2026</a><br/>Подробные протоколы: docs/factory-validation.ru.md · docs/access-control.ru.md · docs/linq-port.md',64,185,1150,19,colors.white)
    text('Данные: <a href="https://data.mendeley.com/datasets/xjmtb22pff/1" color="#B9CDCB">Önal &amp; Dandıl (CC BY 4.0)</a>; <a href="https://github.com/KvCgRv/FPI-Det" color="#B9CDCB">FPI-Det</a>; <a href="https://data.mendeley.com/datasets/6hyrr8typ7/1" color="#B9CDCB">CigDet</a>; URFD; <a href="https://huggingface.co/datasets/raiyaanabdullah/isafety-bench" color="#B9CDCB">iSafetyBench</a>; <a href="https://github.com/pedbrgs/Fire-Detection" color="#B9CDCB">D-Fire</a>. Скриншоты - собственный локальный стенд. Прежний обзор от 23.10 использован как структура, факты обновлены.',64,122,1150,16,colors.HexColor('#B9CDCB'))
    c.showPage()
    if overflow:raise RuntimeError('Slide layout overflow: '+repr(overflow))
    if page!=16:raise AssertionError(page)
    c.save();print(target)


if __name__=='__main__':main()
