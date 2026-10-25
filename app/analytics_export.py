"""Scoped report downloads using the authored XLSX template, with no runtime Node dependency."""
from datetime import datetime, timedelta, timezone
import csv
import io
import math
from pathlib import Path
import re
from typing import Literal
from urllib.parse import quote
import xml.etree.ElementTree as ET
import zipfile

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import Response

from . import analytics, auth, storage as db

router = APIRouter()
TEMPLATE = Path(__file__).parent / 'templates/analytics.xlsx'
MSK = timezone(timedelta(hours=3))
NS = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'
REL = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
PACKAGE_REL = 'http://schemas.openxmlformats.org/package/2006/relationships'
ET.register_namespace('', NS)
ET.register_namespace('r',REL)
Q = lambda name: f'{{{NS}}}{name}'
KINDS = {'no_helmet':'Без каски','danger_zone':'Вход в опасную зону','possible_fall':'Возможное падение',
         'person_down':'Человек на полу','no_vest':'Проверка сигнальной одежды','possible_phone_use':'Возможное использование телефона',
         'possible_smoking':'Возможное курение','fire':'Возможный огонь','smoke':'Возможный дым','fast_motion':'Быстрое перемещение',
         'equipment_proximity':'Сближение с техникой','barrier_crossing':'За ограждением','under_load':'Под грузом',
         'under_conveyor':'Под конвейером','unsafe_load_posture':'Наклон при переноске','no_mask':'Защита лица',
         'no_harness_at_height':'Привязь на высоте'}
STATUSES = {'new':'Ожидает проверки','confirmed':'Подтверждено оператором','false':'Отклонено оператором','forwarded':'Передано мастеру'}
EPISODES = {'open':'Эпизод продолжается','closed':'Эпизод завершён','interrupted':'Наблюдение прервано',
            'legacy_unknown':'Завершение неизвестно','invalid_closed':'Завершение не подтверждено'}
SUMMARY_HEADERS = ['Раздел','Показатель','Количество','Примечание']
EVENT_HEADERS = ['Дата обработки, МСК','Дата события, МСК','Камера','Событие','Зона','Человек в сеансе',
                 'Проверка оператором','Начало в видео','Тревога в видео','Конец в видео','Длительность эпизода',
                 'Состояние эпизода','Кадр','Клип']


def local_date(value):
    stamp = analytics.instant(value)
    return stamp.astimezone(MSK) if stamp else None


def human_date(value):
    stamp = local_date(value)
    return stamp.strftime('%d.%m.%Y, %H:%M:%S') if stamp else 'Неизвестно'


def duration(value):
    value = analytics.number(value)
    if value is None or value < 0: return 'Неизвестно'
    seconds = int(round(value)); hours, seconds = divmod(seconds, 3600); minutes, seconds = divmod(seconds, 60)
    return ' '.join(([f'{hours} ч'] if hours else [])+([f'{minutes} мин'] if minutes else [])+([f'{seconds} с'] if seconds or not hours and not minutes else []))


def video_clock(value):
    """Source offsets retain milliseconds; they are not rounded human durations."""
    value=analytics.number(value)
    if value is None or value < 0:return 'Неизвестно'
    ticks=int(math.floor(value*1000+.5))
    seconds,millis=divmod(ticks,1000);minutes,seconds=divmod(seconds,60);hours,minutes=divmod(minutes,60)
    fraction=f'{millis:03d}'.rstrip('0') or '0'
    return (f'{hours}:{minutes:02d}:' if hours else f'{minutes}:')+f'{seconds:02d}.{fraction}'


def excel_date(value):
    stamp = local_date(value)
    return ((stamp.replace(tzinfo=None)-datetime(1899,12,30)).total_seconds()/86400) if stamp else None


def safe_text(value):
    # OOXML disallows most ASCII control characters. All input remains literal text.
    return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', str(value))[:32767]


def column(index):
    result = ''
    while index:
        index, remainder = divmod(index-1,26); result = chr(65+remainder)+result
    return result


def metadata(report):
    filters = report['filters']; start = local_date(report.get('period_start')); end = local_date(report.get('period_end'))
    unknown=report.get('data_quality',{}).get('source_purpose_events',{}).get('unclassified',0)
    calendar_period=all(re.fullmatch(r'\d{4}-\d{2}-\d{2}',filters.get(key) or '')
                        for key in ('date_from','date_to'))
    if calendar_period:
        # Calendar dates in the UI include the whole last Moscow day.
        period=start.strftime('%d.%m.%Y')+' — '+(end-timedelta(microseconds=1)).strftime('%d.%m.%Y')+' включительно · МСК (UTC+3)'
    elif start:
        # Rolling and explicit ISO periods are instants, not whole calendar days.
        period=start.strftime('%d.%m.%Y %H:%M:%S')+' — '+end.strftime('%d.%m.%Y %H:%M:%S')+' (конец не включён; МСК UTC+3)'
    else:
        period='Всё время · по состоянию на '+(end.strftime('%d.%m.%Y %H:%M:%S') if end else human_date(report['generated_at']))+' · МСК (UTC+3)'
    return [
        ['Период',period],
        ['Дата отбора','Время источника (LIVE — получение кадра сервером), МСК (UTC+3)' if filters['time_basis']=='recording' else 'Дата обработки, МСК (UTC+3)'],
        ['Сформирован',excel_date(report['generated_at'])],
        ['Камеры',filters.get('camera') or 'Все доступные камеры'],
        ['Отбор',('Вся история сценариев' if filters.get('include_experimental') else 'Разрешённые рабочие сценарии')+'. '+
            ('Все статусы' if filters.get('status','all')=='all' else 'Статус: '+STATUSES.get(filters['status'],filters['status']))+'. '+
            ('Учебные включены.' if filters.get('include_examples') else 'Явные учебные исключены.')+
            (f' Без отметки назначения: {unknown}.' if unknown else '')]
    ]


def summary_rows(report):
    k = report['kpis']; out = []
    for key,title in [('automatic_events','Автоматические события'),('pending','Ожидают проверки'),
                      ('confirmed','Подтверждены оператором'),('rejected','Отклонены оператором'),
                      ('unresolved_episodes','Завершение эпизода неизвестно')]:
        out.append(['Итоги',title,k.get(key,0),'За выбранный период и по выбранным фильтрам'])
    for kind,count in sorted(report['breakdowns']['by_kind'].items()):
        out.append(['Типы событий',KINDS.get(kind,kind),count,'Автоматические события; проверка оператором указана отдельно'])
    for state,count in sorted(report['breakdowns']['by_status'].items()):
        out.append(['Проверка оператором',STATUSES.get(state,state),count,''])
    for purpose,count in report.get('data_quality',{}).get('source_purpose_events',{}).items():
        title={'educational':'Учебные','declared_production':'Отмечены как рабочие','unclassified':'Назначение источника не указано'}.get(purpose,purpose)
        out.append(['Назначение источников',title,count,'Неотмеченные источники не считаются подтверждёнными рабочими камерами'])
    for row in report['cameras']:
        out.append(['Камеры',row['camera'],row['events'],f"Ожидают: {row['pending']}; подтверждены: {row['confirmed']}; отклонены: {row['rejected']}"])
        for kind,count in sorted(row.get('by_kind',{}).items()):
            out.append(['Камеры / типы',row['camera']+' · '+KINDS.get(kind,kind),count,'Те же фильтры периода и статуса'])
    for date,count in report['breakdowns'].get('by_time_day',report['breakdowns']['by_processing_day']).items():
        stamp=datetime.fromisoformat(date).strftime('%d.%m.%Y')
        source_time=report['filters']['time_basis']=='recording'
        out.append(['Дни событий' if source_time else 'Дни обработки',stamp,count,
            'Время источника, МСК; LIVE — получение кадра сервером' if source_time else 'Дата обработки, не дата съёмки'])
    episodes=report['episodes']; closed=episodes['closed_count']
    out += [['Эпизоды','Завершённые интервалы',closed,'Незавершённые эпизоды не считаются нулевыми'],
            ['Эпизоды','Сумма завершённых интервалов',None,duration(episodes['closed_interval_total_s']) if closed else 'Нет завершённых интервалов'],
            ['Эпизоды','Медиана завершённого интервала',None,duration(episodes['closed_interval_median_s'])],
            ['Разбор','Медиана до первого решения',None,duration(report['review']['median_first_decision_s'])],
            ['Метод','Подсчёт',None,'Повторный анализ одной записи может создать отдельные события. Сумма интервалов не является временем простоя предприятия.']]
    if not report.get('can_drilldown'):
        out.append(['Доступ','Детализация',None,'Эта роль получает агрегаты. Список отдельных событий недоступен.'])
    return out


def event_rows(report, *, xlsx):
    result=[]
    for e in report.get('events',[]):
        numeric=lambda key: (analytics.number(e.get(key))/86400 if analytics.number(e.get(key)) is not None else None) if xlsx else duration(e.get(key)) if key=='duration_s' else video_clock(e.get(key))
        row=[excel_date(e.get('created')) if xlsx else human_date(e.get('created')),
             excel_date(e.get('occurred_at')) if xlsx else human_date(e.get('occurred_at')),
             e['camera'],KINDS.get(e['kind'],e['kind']),e.get('zone') or 'Весь кадр',e.get('track_id'),
             STATUSES.get(e['status'],e['status']),numeric('start_s'),numeric('alert_s'),numeric('end_s'),numeric('duration_s'),
             EPISODES.get(e['episode_state'],'Завершение неизвестно'),e.get('image') or '',e.get('clip') or '']
        result.append(row)
    return result


def _sheet_xml(raw, title, meta, headers, rows, links=None):
    sheet=ET.fromstring(raw); source=sheet.find(Q('sheetData'))
    styles={c.get('r'):c.get('s','0') for r in source for c in r}
    data=ET.Element(Q('sheetData'))
    def add_row(n,values,kind='body'):
        row=ET.SubElement(data,Q('row'),r=str(n))
        if n==9: row.set('ht','33'); row.set('customHeight','1')
        elif n>=10: row.set('ht','27'); row.set('customHeight','1')
        for i,value in enumerate(values,1):
            address=f'{column(i)}{n}'
            style=styles.get(f'{column(i)}10','0') if kind=='body' else styles.get(address,'0')
            cell=ET.SubElement(row,Q('c'),r=address,s=style)
            if value is None: continue
            if isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value):
                ET.SubElement(cell,Q('v')).text=str(value)
            else:
                cell.set('t','inlineStr'); text=ET.SubElement(ET.SubElement(cell,Q('is')),Q('t'))
                text.set('{http://www.w3.org/XML/1998/namespace}space','preserve'); text.text=safe_text(value)
    add_row(2,[title],'title')
    for n,values in enumerate(meta,3): add_row(n,values,'meta')
    add_row(9,headers,'header')
    for n,values in enumerate(rows or [[None]*len(headers)],10): add_row(n,values)
    index=list(sheet).index(source); sheet.remove(source); sheet.insert(index,data)
    if links:
        hyperlinks=ET.Element(Q('hyperlinks'))
        for address,identifier in links.items():
            ET.SubElement(hyperlinks,Q('hyperlink'),ref=address,attrib={f'{{{REL}}}id':identifier})
        table_parts=sheet.find(Q('tableParts'))
        sheet.insert(list(sheet).index(table_parts) if table_parts is not None else len(sheet),hyperlinks)
    final=f'A9:{column(len(headers))}{max(10,9+len(rows))}'
    dim=sheet.find(Q('dimension'))
    if dim is not None: dim.set('ref',f'A1:{column(len(headers))}{max(10,9+len(rows))}')
    return ET.tostring(sheet,encoding='utf-8',xml_declaration=True),final


def xlsx_bytes(report):
    meta=metadata(report); matrices=[summary_rows(report),event_rows(report,xlsx=True)]
    if not report.get('can_drilldown'): meta_detail=[*meta[:4],['Отбор','Только агрегаты: отдельные события недоступны для этой роли.']]
    elif not matrices[1]: meta_detail=[*meta[:4],['Отбор','За выбранный период по выбранным фильтрам событий нет.']]
    else: meta_detail=meta
    output=io.BytesIO()
    with zipfile.ZipFile(TEMPLATE) as template,zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as archive:
        sheets={}; refs={}
        # Native hyperlinks preserve complete authenticated media URLs without
        # long clipped text or executable HYPERLINK formulas in cells.
        links={}; relations=ET.fromstring(template.read('xl/worksheets/_rels/sheet2.xml.rels'))
        for n,row in enumerate(matrices[1],10):
            for i,label in ((12,'Открыть кадр'),(13,'Открыть клип')):
                value=row[i]
                if isinstance(value,str) and value.startswith(('http://','https://')):
                    identifier=f'rIdProof{len(links)+1}'; links[f'{column(i+1)}{n}']=identifier
                    ET.SubElement(relations,f'{{{PACKAGE_REL}}}Relationship',Id=identifier,Type=REL+'/hyperlink',Target=value,TargetMode='External')
                    row[i]=label
        sheets['xl/worksheets/_rels/sheet2.xml.rels']=ET.tostring(relations,encoding='utf-8',xml_declaration=True)
        for index,(title,headers,values) in enumerate(zip(('Статистика безопасности','Происшествия'),(SUMMARY_HEADERS,EVENT_HEADERS),matrices),1):
            name=f'xl/worksheets/sheet{index}.xml'
            sheets[name],refs[index]=_sheet_xml(template.read(name),title,meta if index==1 else meta_detail,headers,values,links if index==2 else None)
        for item in template.infolist():
            data=sheets.get(item.filename)
            if data is None:
                data=template.read(item)
                if re.fullmatch(r'xl/tables/table[12]\.xml',item.filename):
                    index=int(Path(item.filename).stem[-1]); table=ET.fromstring(data); table.set('ref',refs[index])
                    af=table.find(Q('autoFilter'))
                    if af is not None: af.set('ref',refs[index])
                    data=ET.tostring(table,encoding='utf-8',xml_declaration=True)
            archive.writestr(item.filename,data)
    return output.getvalue()


def csv_bytes(report):
    stream=io.StringIO(newline=''); writer=csv.writer(stream,delimiter=';')
    def write(values):
        cells=[]
        for value in values:
            text='' if value is None else safe_text(value)
            if isinstance(value,str) and text.lstrip().startswith(('=','+','-','@')): text="'"+text
            cells.append(text)
        writer.writerow(cells)
    write(['Статистика безопасности'])
    for key,value in metadata(report): write([key,human_date(report['generated_at']) if key=='Сформирован' else value])
    write([]); write(SUMMARY_HEADERS)
    for row in summary_rows(report): write(row)
    if report.get('can_drilldown'):
        write([]); write(EVENT_HEADERS)
        for row in event_rows(report,xlsx=False): write(row)
    return ('\ufeff'+stream.getvalue()).encode('utf-8')


@router.get('/api/analytics/export')
def export(request: Request, format: Literal['xlsx','csv']='xlsx', days: Literal['7','30','all']='7', camera: str='',
           time_basis: Literal['processing','recording']='processing', include_experimental: bool=False, include_examples: bool=False,
           date_from: str | None=Query(None,alias='from'), date_to: str | None=Query(None,alias='to'),
           status: Literal['all','new','forwarded','confirmed','false']='all'):
    user=auth.require(request,'analytics.read')
    if len(camera)>200: raise HTTPException(422,'Название камеры слишком длинное')
    try:
        report=analytics.build_analytics(days=None if days=='all' else int(days),camera=camera,time_basis=time_basis,
            include_experimental=include_experimental,include_examples=include_examples,date_from=date_from,date_to=date_to,
            allow_events='events.read' in user['permissions'],status=status)
    except analytics.pattern_policy.PolicyError as exc:raise HTTPException(503,str(exc)) from None
    except ValueError as exc: raise HTTPException(422,str(exc)) from None
    if report.get('can_drilldown') and 'evidence.read' in user['permissions']:
        # Enrich only already-filtered report rows using the same tenant/camera scope.
        evidence={event['id']:event for event in db.events()}
        def proof_url(value):
            return str(request.base_url).rstrip('/')+'/media/'+quote(value.replace('\\','/'),safe='/') if value else ''
        for row in report['events']:
            original=evidence.get(row['id'],{})
            row.update(track_id=original.get('track_id'),
                       image=proof_url(original.get('image')),clip=proof_url(original.get('clip')))
    content=xlsx_bytes(report) if format=='xlsx' else csv_bytes(report)
    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' if format=='xlsx' else 'text/csv; charset=utf-8'
    # Parsed calendar dates only; raw user strings never enter attachment headers.
    start=local_date(report.get('period_start')); end=local_date(report.get('period_end'))
    first=start.strftime('%Y-%m-%d') if start else 'all'
    last=(end-timedelta(microseconds=1)).strftime('%Y-%m-%d') if end else 'now'
    filename=f'linq-statistics_{first}_{last}.{format}'
    return Response(content,media_type=mime,headers={'Content-Disposition':f'attachment; filename="{filename}"',
                                                  'Cache-Control':'no-store'})
