// Regenerate the styled workbook template with the bundled Artifact Tool.
// Run from a task folder whose node_modules points to the bundled runtime.
import fs from 'node:fs/promises';
import path from 'node:path';
import { createRequire } from 'node:module';
import { pathToFileURL } from 'node:url';
const require = createRequire(path.join(process.cwd(), 'package.json'));
const { Workbook, SpreadsheetFile } = await import(pathToFileURL(require.resolve('@oai/artifact-tool')).href);
const destination = process.argv[2];
if (!destination) throw new Error('Pass output analytics.xlsx path');
const wb = Workbook.create();
const demoDate = (Date.parse('2026-10-25T03:15:00Z')+3*3600000)/86400000+25569;
const definitions = [
  {name:'Сводка', headers:['Раздел','Показатель','Количество','Примечание'], widths:[27,44,17,76],
   sample:['Происшествия','События за выбранный период',12,'Учёт автоматических событий, без учебных примеров']},
  {name:'Происшествия', headers:['Дата обработки, МСК','Дата события, МСК','Камера','Событие','Зона','Человек в сеансе','Проверка оператором','Начало в видео','Тревога в видео','Конец в видео','Длительность эпизода','Состояние эпизода','Кадр','Клип'],
   widths:[25,25,30,32,26,20,31,22,22,22,29,30,20,20],
   sample:[demoDate,null,'Линия сборки 1','Без каски','Рабочее место',7,'Ожидает проверки',.5/86400,2/86400,null,null,'Наблюдение прервано','','']}
];
for (const [index,d] of definitions.entries()) {
  const sheet=wb.worksheets.add(d.name); sheet.showGridLines=false;
  sheet.tabColor=index?'#52766B':'#173F41';
  const end=String.fromCharCode(64+d.headers.length);
  sheet.getRange(`A1:${end}12`).format.font={name:'Arial',size:10,color:'#243B3A'};
  sheet.getRange(`A1:${end}12`).format.verticalAlignment='center';
  d.widths.forEach((width,c)=>sheet.getRange(`${String.fromCharCode(65+c)}1`).format.columnWidth=width);
  sheet.getRange('A2').values=[[index?'Происшествия':'Статистика безопасности']];
  sheet.getRange('A2').format.font={name:'Arial',size:15,bold:true,color:'#173F41'};
  sheet.getRange('A3').values=[['Период']]; sheet.getRange('B3').values=[['25.10.2026 — 25.10.2026']];
  sheet.getRange('A4').values=[['Дата отбора']]; sheet.getRange('B4').values=[['Дата обработки, московское время (UTC+3)']];
  sheet.getRange('A5').values=[['Сформирован']]; sheet.getRange('B5').values=[[demoDate]];
  sheet.getRange('B5').setNumberFormat('dd.mm.yyyy hh:mm');
  sheet.getRange('A6').values=[['Камеры']]; sheet.getRange('B6').values=[['Все доступные камеры']];
  sheet.getRange('A7').values=[['Отбор']]; sheet.getRange('B7').values=[['Основные сценарии. Учебные примеры исключены.']];
  sheet.getRange(`A9:${end}9`).values=[d.headers];
  sheet.getRange(`A9:${end}9`).format={fill:'#173F41',font:{name:'Arial',size:10,bold:true,color:'#FFFFFF'},wrapText:true,horizontalAlignment:'center',rowHeight:33};
  sheet.getRange(`A10:${end}10`).values=[d.sample];
  sheet.getRange(`A10:${end}10`).format.rowHeight=30;
  sheet.tables.add(`A9:${end}10`,true,index?'IncidentDetails':'IncidentSummary');
  if (!index) sheet.getRange('C10:C12').setNumberFormat('#,##0');
  else {
    sheet.getRange('A10:B12').setNumberFormat('dd.mm.yyyy hh:mm:ss');
    sheet.getRange('F10:F12').setNumberFormat('0');
    sheet.getRange('H10:K12').setNumberFormat('[h]:mm:ss.0');
    sheet.getRange('M10:N12').format.font={name:'Arial',size:10,color:'#1E6393'};
  }
  sheet.freezePanes.freezeRows(9); if(index)sheet.freezePanes.freezeColumns(3);
}
wb.recalculate();
await fs.mkdir(path.dirname(destination),{recursive:true});
for (const d of definitions) {
  const preview=await wb.render({sheetName:d.name,range:d.name==='Сводка'?'A1:D12':'A1:G12',scale:1,format:'png'});
  await fs.writeFile(path.join(process.cwd(),d.name==='Сводка'?'summary-template.png':'events-template.png'),new Uint8Array(await preview.arrayBuffer()));
}
const inspect=await wb.inspect({kind:'table',range:'Сводка!A9:D10',include:'values',tableMaxRows:2,tableMaxCols:4,maxChars:1500});
console.log(inspect.ndjson);
await (await SpreadsheetFile.exportXlsx(wb)).save(destination);
console.log('Saved template',destination);
