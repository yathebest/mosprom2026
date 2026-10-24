"""Build the dated Russian engineering decision record from reviewed reports."""
from datetime import datetime,timedelta,timezone
import html
from pathlib import Path
import re

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,KeepTogether

ROOT=Path(__file__).resolve().parents[1]
INK=colors.HexColor('#173F42');MUTED=colors.HexColor('#52676B');PALE=colors.HexColor('#EEF4EB')


def main():
    for name,file in [('Text','segoeui.ttf'),('Bold','segoeuib.ttf'),('Italic','segoeuii.ttf'),('Code','consola.ttf')]:
        pdfmetrics.registerFont(TTFont(name,str(Path('C:/Windows/Fonts')/file)))
    pdfmetrics.registerFontFamily('Text',normal='Text',bold='Bold',italic='Italic',boldItalic='Bold')
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='BodyRu',fontName='Text',fontSize=10.1,leading=14.5,textColor=INK,spaceAfter=7))
    styles.add(ParagraphStyle(name='SmallRu',parent=styles['BodyRu'],fontSize=8.5,leading=12,spaceAfter=5,textColor=MUTED))
    styles.add(ParagraphStyle(name='TitleRu',parent=styles['BodyRu'],fontName='Bold',fontSize=28,leading=34,spaceAfter=16))
    styles.add(ParagraphStyle(name='H1Ru',parent=styles['BodyRu'],fontName='Bold',fontSize=18,leading=23,spaceBefore=14,spaceAfter=10,keepWithNext=True))
    styles.add(ParagraphStyle(name='H2Ru',parent=styles['BodyRu'],fontName='Bold',fontSize=13,leading=18,spaceBefore=12,spaceAfter=7,keepWithNext=True))
    styles.add(ParagraphStyle(name='CodeRu',parent=styles['SmallRu'],fontName='Code',fontSize=8,leading=11,backColor=PALE,borderPadding=8))
    styles.add(ParagraphStyle(name='CellRu',parent=styles['BodyRu'],fontSize=9,leading=12,spaceAfter=0))
    styles.add(ParagraphStyle(name='HeadCellRu',parent=styles['CellRu'],fontName='Bold',textColor=colors.white))

    def rich(s):
        s=s.replace('\u2011','-').replace('\u2013','-').replace('\u2014','-')
        s=html.escape(s)
        s=re.sub(r'\[([^\]]+)\]\((https?://[^)]+)\)',r'<a href="\2" color="#216A72">\1</a>',s)
        s=re.sub(r'\*\*(.+?)\*\*',r'<b>\1</b>',s)
        s=re.sub(r'`([^`]+)`',r'<font name="Code">\1</font>',s)
        return s
    def p(s,style='BodyRu'):return Paragraph(rich(s),styles[style])
    story=[]
    def table(rows,widths=None):
        n=len(rows[0]);widths=widths or ([88*mm]+[(174-88)*mm/(n-1)]*(n-1) if n>1 else [174*mm])
        cells=[[p(cell,'HeadCellRu' if i==0 else 'CellRu') for cell in row] for i,row in enumerate(rows)]
        t=Table(cells,colWidths=widths,repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),INK),('VALIGN',(0,0),(-1,-1),'TOP'),
            ('LEFTPADDING',(0,0),(-1,-1),9),('RIGHTPADDING',(0,0),(-1,-1),9),
            ('TOPPADDING',(0,0),(-1,-1),8),('BOTTOMPADDING',(0,0),(-1,-1),8),
            ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,PALE]),
            ('LINEBELOW',(0,0),(-1,0),.5,INK),('LINEBELOW',(0,1),(-1,-1),.3,colors.HexColor('#DCE6DE'))]))
        return t

    now=datetime.now(timezone(timedelta(hours=3)))
    story += [p('LINQ SAFETY  /  ИНЖЕНЕРНЫЙ ЖУРНАЛ','SmallRu'),Spacer(1,10*mm),
        p('Решения и результаты<br/>за 24 октября 2026','TitleRu')]
    # The title uses a deliberate line break, not user-supplied markup.
    story[-1]=Paragraph('Решения и результаты<br/>за 24 октября 2026',styles['TitleRu'])
    story += [p(f'Срез состояния: {now:%d.%m.%Y %H:%M} МСК. Документ фиксирует сегодняшнюю работу; не заменяет живой журнал Kaggle.','SmallRu'),
        p('Цель на сегодня: проверить решение на реальных рабочих записях, устранить пропуски людей и СИЗ, улучшить телефон и курение, ускорить LIVE и принимать новые веса только после практического сравнения.'),
        table([['Решили','Что сделано / текущий статус'],
            ['Сохранить совместимую основу','YOLO12m для людей и кандидатов объектов; ResNet50 для проверок СИЗ. Сильные рабочие веса не заменяются автоматически.'],
            ['Разделить видео и нейросети','Захват, предпросмотр, запись, люди и медленные сценарии разделены. Очереди держат последний кадр.'],
            ['Проверять дальние планы','Добавлены исходные участки кадра, уточнение позы и нейтральные предложения головы/корпуса.'],
            ['Улучшать на данных, а не порогами','Подготовлены заводские проверки, FPI-Det и CigDet. Несколько ухудшающих кандидатов отклонены.'],
            ['Запустить дообучение','Сигареты: 13 эпох завершены, ONNX и PT сохранены локально. Phone/head перезапущен отдельно: входные данные проверены, реальные батчи видны в общем журнале.'],
            ['Сохранять восстановление','Чекпойнт с оптимизатором каждой эпохи + лучшие веса в результатах Kaggle. Внешний API-бэкап отключён по решению владельца.']],[51*mm,123*mm]),
        Spacer(1,5*mm),p('Статус качества: улучшения подтверждены на ограниченных отладочных выборках. Метрики всех заводских событий выше 0,95 не получены. Самые серьёзные оставшиеся риски - падения и мелкие предметы на дальних камерах.','SmallRu'),PageBreak()]

    story += [p('Выбранная стратегия моделей','H1Ru'),
        p('1. Людей ищем ансамблем дообученного и исходного COCO YOLO12m. Дополнительные масштабы и участки дополняют основной проход. Это важнее безусловного перехода на более крупные веса: простая замена масштаба уже дала регрессии.'),
        p('2. Каски и жилеты проверяем в два этапа: детектор предлагает область, ResNet50 проверяет её класс. Если детектор пропустил область, используем измеренные точки лица/корпуса. Предложение области само по себе не означает нарушение.'),
        p('3. Для телефона обучаем YOLO12m на телефон и голову. Сохраняем 80 выходов для переноса предобученных весов: телефон остаётся классом 67, область головы занимает класс 0. Такую модель нельзя поставить вместо модели людей: её класс 0 уже означает голову.'),
        p('4. Для курения оставляем проверку видимого предмета у конкретного человека и исходный участок головы/рук. Жест руки или дым не считаем доказательством сигареты. Для маленького человека честно ограничиваем уверенность решения.'),
        p('5. Для огня добавлен проверенный D-Fire YOLOv5s только по классу пламени. Старый огонь/дым сохранён. Новые срабатывания дыма на паре/пыли не принимаем автоматически.'),
        p('6. Падения: существующие ResNet18 и GRU продолжаем проверять на рабочих сценах. Высокий лабораторный clip F1 оказался недостаточным. Короткие изменения правил без практического выигрыша не устанавливаем.'),
        p('Совместимость с LinQ','H2Ru'),
        p('Семейства YOLO12m, YOLOv5s и YOLOv8n-pose перечислены в предоставленном SDK. Предыдущая компиляционная проверка YOLO12m относится к статическому входу 640; новые веса и вход 960 требуют отдельной компиляции, калибровки и запуска на оборудовании. Работа на локальной RTX 4060 не является проверкой производительности LinQ.'),
        p('Рабочее правило: сначала сравнение кандидата и исходной модели на одних данных, затем контроль обычной работы/ложных тревог, затем установка с резервной копией. Автоматической подмены веса после обучения нет.','SmallRu'),PageBreak()]

    # The reviewed repository report carries detailed methods, controls and caveats.
    source=(ROOT/'docs/factory-validation.ru.md').read_text(encoding='utf-8')
    lines=source.splitlines();i=0;code=False;code_lines=[]
    while i<len(lines):
        line=lines[i].strip();i+=1
        if line.startswith('```'):
            if not code:code=True;code_lines=[]
            else:
                story.append(Paragraph('<br/>'.join(html.escape(s) for s in code_lines),styles['CodeRu']));story.append(Spacer(1,4*mm));code=False
            continue
        if code:code_lines.append(line);continue
        if not line:continue
        if line.startswith('|'):
            rows=[[c.strip() for c in line.strip('|').split('|')]]
            while i<len(lines) and lines[i].strip().startswith('|'):
                cells=[c.strip() for c in lines[i].strip().strip('|').split('|')];i+=1
                if all(re.fullmatch(r'[:\- ]+',c) for c in cells):continue
                rows.append(cells)
            story.extend([table(rows),Spacer(1,4*mm)]);continue
        if line.startswith('# '):
            story.append(p('Подробный журнал практических проверок','H1Ru'));continue
        if line.startswith('## '):story.append(p(line[3:],'H2Ru'));continue
        if line.startswith('- '):story.append(p('• '+line[2:]));continue
        story.append(KeepTogether([p(line)]))

    story += [PageBreak(),p('Дополнительные выводы и порядок продолжения','H1Ru'),
        p('Сравнение альтернативного архива весов','H2Ru'),
        p('Архив linq-trained-models.zip проверен отдельно от установленных ночных моделей. В нём оказались результаты короткого smoke-запуска, а не полноценный более сильный набор. Метрики PPE-детекторов и классификаторов были существенно хуже; рабочие веса не заменены. Сравнение и ограничения протокола сохранены в docs/alternative-weights-comparison.json.'),
        p('Исследование по промышленной безопасности','H2Ru'),
        p('Изучена работа Action Recognition based Industrial Safety Violation Detection (arXiv:2412.05531): SlowFast для действий и YOLOv9 для СИЗ. Из неё берём идею сочетания действий, экипировки и контекста камеры. Перенос SlowFast в LinQ не выполнен. Высокая метрика отдельного детектора касок не равна качеству всей цепочки: в работе итоговый F1 для всех PPE-правил около 0,54. Это поддерживает выбранную проверку именно событий, а не красивой цифры классификатора.'),
        p('Что проверено программно','H2Ru'),
        p('Перед внедрением новых состояний пройдены 119 Python-тестов, 26 JavaScript-тестов и 4 теста наблюдаемости обучения. Проверяются пропуски/неопределённость, свежесть наблюдений, смены ID, повторные эпизоды, координаты фрагментов, независимость видео от медленной модели, H264-клипы и сопровождение рамок без права подтверждать тревогу. Эти тесты подтверждают работу логики, а не точность нейросети на всех камерах.'),
        p('Вечернее уточнение: обучение и сохранённые модели','H2Ru'),
        p('Полный дочерний лог старого phone/head показал одну завершённую эпоху и часть второй. Общий журнал этого не отображал, а предпросмотр файла Kaggle обрезал текст. Поэтому отсутствие эпох в прежнем интерфейсе не доказывало остановку вычислений. Новый отдельный запуск 361832495 выполняется с исходного YOLO12m по решению владельца; сигареты повторно не обучаются.'),
        p('В новый ноутбук добавлены этапы копирования данных, загрузки модели, проверки датасета, батчей, валидации, сохранения и экспорта. Каждые 20 секунд выводится ограниченный фрагмент дочернего лога и компактный heartbeat. Train и validation копируются локально; test при обучении не читается. Оптимизатор сохраняется по эпохам в результатах Kaggle; уничтожение виртуальной машины внешним бэкапом не покрывается.'),
        p('Сигаретный кандидат проверен на 89 validation-изображениях при confidence 0,5 и IoU 0,5. Исходная модель: Precision 0,837 / Recall 0,809 / F1 0,823. Новый кандидат: 0,951 / 0,438 / 0,600. Он реже ошибается среди своих находок, но пропускает больше объектов, поэтому в приложение не установлен. При уменьшении изображения в четыре раза Recall кандидата 0,112 против 0,247 у исходной модели. Это синтетическая проверка мелких объектов, не заводские события.'),
        p('Решение по состояниям человека и дублям','H2Ru'),
        p('Для каждого track ID и паттерна вводится отдельное подтверждённое состояние: неизвестно, нарушение, безопасный признак. Окна зависят от паттерна; для перехода требуется не менее 75% времени окна и несколько разных свежих наблюдений. Неизвестность, прогноз рамки и повтор того же timestamp не доказывают ни нарушение, ни его устранение.'),
        p('Вход в подтверждённое нарушение открывает один эпизод и одну тревогу. Повторные кадры обновляют состояние без новых тревог. Подтверждённое устранение закрывает тот же эпизод, сохраняет интервал и кадр завершения. Пропажа человека, конец видео или остановка камеры оставляют завершение неподтверждённым. Короткий клип является фрагментом доказательства, а не записью всей длительности нарушения. Фактический статус внедрения фиксируется в отчёте проверки состояний.'),
        p('Исправления интерфейса','H2Ru'),
        p('Добавлены переходы по верхней навигации и шагам мастера, удаление любой вершины и отмена редактирования зоны, перенумерация стандартных имён, конкретное объяснение зоны высоты, читаемый аудит и местный часовой пояс. Исправлена бесконечная проверка оборудования; добавлены RU/EN и пояснения чувствительности. Переходы и редактор проверены кликами. Различение дыма и пыли моделью пока не подтверждено.'),
        p('Порядок дальнейших действий','H2Ru'),
        p('1. Дождаться завершения отдельного phone/head и проверить сохранённые оптимизаторные чекпойнты, лучшие веса и ONNX. Наблюдаемость уже исправлена: общий журнал показывает текущий этап и счётчик реальных батчей, а не только завершённые эпохи.'),
        p('2. После обучения сравнить новых кандидатов на validation при фиксированных условиях. Только прошедшие проверку модели оценивать на отложенных источниках и подключать в приложение. Для телефона отдельно проверять предмет, привязку к руке, начало эпизода и повторные тревоги.'),
        p('3. Для падений собрать пригодные именно для задачи видимые интервалы падения и обычной работы. Не превращать целый аварийный клип в положительную метку каждой секунды. Включить наклоны, приседания, сидение и перекрытия как сложные отрицательные примеры; разделять исходные видео/камеры.'),
        p('4. Повторить UI-сценарии: камера/запись, тревога, сохранение и воспроизведение клипа, отсутствие повтора продолжающегося события. Считать задержку от исходного кадра, ложные тревоги на камеро-час и долю unknown, а не только FPS.'),
        p('5. После локального контроля - компиляция и измерение на LinQ. Доступ через VPN выполняется отдельно, поскольку подключение прерывает связь в текущем рабочем окружении.'),
        p('Основные материалы','H2Ru'),
        p('[Репозиторий проекта](https://github.com/yathebest/mosprom2026) · [Текущий phone/head](https://www.kaggle.com/code/yathebest/linq-phone-head-fresh-logged-2026?scriptVersionId=361832495) · [Сохранённые сигареты](https://www.kaggle.com/code/yathebest/linq-factory-objects-training-2026/output?scriptVersionId=361806401) · [Входные данные](https://www.kaggle.com/datasets/yathebest/linq-factory-objects-2026)','SmallRu'),
        p('[FPI-Det](https://github.com/KvCgRv/FPI-Det) · [CigDet](https://data.mendeley.com/datasets/6hyrr8typ7/1) · [Заводские записи](https://data.mendeley.com/datasets/xjmtb22pff/1) · [iSafetyBench](https://huggingface.co/datasets/raiyaanabdullah/isafety-bench) · [D-Fire модели](https://github.com/pedbrgs/Fire-Detection) · [Исследование](https://arxiv.org/abs/2412.05531)','SmallRu')]

    target=ROOT/'output/pdf/LinQ-decisions-2026-10-24.pdf';target.parent.mkdir(parents=True,exist_ok=True)
    def chrome(canvas,doc):
        canvas.saveState();w,h=doc.pagesize
        canvas.setStrokeColor(colors.HexColor('#CBD9CE'));canvas.line(18*mm,h-15*mm,w-18*mm,h-15*mm)
        canvas.setFont('Text',8);canvas.setFillColor(MUTED)
        canvas.drawString(18*mm,h-12*mm,'LINQ SAFETY  /  24.10.2026')
        canvas.drawString(18*mm,12*mm,'Рабочий журнал. Отладочные измерения не равны промышленной сертификации.')
        canvas.drawRightString(w-18*mm,12*mm,str(doc.page));canvas.restoreState()
    doc=SimpleDocTemplate(str(target),pagesize=(210*mm,297*mm),leftMargin=18*mm,rightMargin=18*mm,
        topMargin=22*mm,bottomMargin=23*mm,title='LinQ Safety - решения и результаты 24 октября 2026',author='Команда LinQ Safety')
    doc.build(story,onFirstPage=chrome,onLaterPages=chrome)
    print(target)


if __name__=='__main__':main()
