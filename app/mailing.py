"""SMTP delivery and an explicitly enabled local preview transport for demos."""
from dataclasses import dataclass
from email.message import EmailMessage
import json
import os
import re
import smtplib
import ssl
from urllib.parse import urlsplit
import uuid

from fastapi import HTTPException
from . import storage as db


def email_address(value):
    value=value.strip().lower()
    if len(value)>120 or not re.fullmatch(r'[a-z0-9][a-z0-9_.+\-]*@[a-z0-9](?:[a-z0-9\-]*[a-z0-9])?(?:\.[a-z0-9](?:[a-z0-9\-]*[a-z0-9])?)+',value):
        raise HTTPException(422,'Укажите email без пробелов; максимальная длина — 120 символов')
    return value


def settings():
    path=db.DATA/'mail.local.json'
    config={'mode':'preview','public_url':'http://127.0.0.1:8765'}
    if path.exists():
        try:
            config=json.loads(path.read_text(encoding='utf-8-sig'))
            if not isinstance(config,dict):raise ValueError()
        except (ValueError,OSError):raise HTTPException(503,'Некорректный локальный файл настройки почты') from None
    for key in ('mode','host','port','security','username','password','from_address','public_url'):
        env=os.getenv('SAFETY_MAIL_'+key.upper())
        if env is not None:config[key]=env
    return config


@dataclass
class Delivery:
    config: dict
    public_url: str
    mode: str

    def send(self,email,enterprise,link,expires_at):
        subject='LinQ Safety: приглашение администратора предприятия'
        text=(f'Вас пригласили администрировать предприятие «{enterprise}» в LinQ Safety.\n\n'
              f'Email аккаунта: {email}\n\nЗарегистрируйте аккаунт по ссылке:\n{link}\n\n'
              'Ссылка действует 60 секунд с момента создания и только один раз. '
              'Повторная отправка заменяет предыдущую ссылку. Если срок истёк, '
              'попросите администратора платформы отправить новое приглашение.\n'
              'Если вы не ожидали приглашения, не переходите по ссылке.\n')
        if self.mode=='preview':
            key=uuid.uuid4().hex;folder=db.DATA/'mail-preview';folder.mkdir(exist_ok=True)
            (folder/(key+'.json')).write_text(json.dumps({'to':email,'subject':subject,'text':text,'link':link,'expires_at':expires_at},ensure_ascii=False),encoding='utf-8')
            # Keep only a bounded number of local demonstration messages.
            for old in sorted(folder.glob('*.json'),key=lambda p:p.stat().st_mtime,reverse=True)[50:]:old.unlink(missing_ok=True)
            return {'delivery':'preview','preview_id':key}
        message=EmailMessage();message['Subject']=subject;message['From']=self.config['from_address'];message['To']=email;message.set_content(text)
        try:
            context=ssl.create_default_context()
            if self.config['security']=='ssl':
                smtp=smtplib.SMTP_SSL(self.config['host'],self.config['port'],timeout=10,context=context)
            else:
                smtp=smtplib.SMTP(self.config['host'],self.config['port'],timeout=10)
            with smtp:
                smtp.ehlo()
                if self.config['security']=='starttls':smtp.starttls(context=context);smtp.ehlo()
                if self.config.get('username'):smtp.login(self.config['username'],self.config['password'])
                rejected=smtp.send_message(message)
                if rejected:raise smtplib.SMTPRecipientsRefused(rejected)
        except (OSError,smtplib.SMTPException):
            # SMTP errors may contain credentials/addresses. Never return raw errors.
            raise HTTPException(502,'Почтовый сервер не подтвердил отправку. Приглашение отозвано; проверьте SMTP и отправьте заново.') from None
        return {'delivery':'smtp'}


def prepare():
    config=settings();mode=config.get('mode','smtp')
    if mode not in ('smtp','preview'):raise HTTPException(503,'Неизвестный режим почты')
    public_url=config.get('public_url','http://127.0.0.1:8765').rstrip('/')
    parsed=urlsplit(public_url)
    if parsed.scheme not in ('http','https') or not parsed.hostname or parsed.username or parsed.password or parsed.path not in ('','/') or parsed.query or parsed.fragment:
        raise HTTPException(503,'Настройте адрес сайта SAFETY_MAIL_PUBLIC_URL без пути и реквизитов доступа')
    if parsed.scheme!='https' and parsed.hostname not in ('localhost','127.0.0.1','::1'):
        raise HTTPException(503,'Для внешних приглашений нужен HTTPS-адрес сайта')
    if mode=='smtp':
        if not config.get('host') or not config.get('from_address'):raise HTTPException(503,'SMTP не настроен. Заполните data/mail.local.json по образцу mail.example.json.')
        if config.get('security') not in ('ssl','starttls'):raise HTTPException(503,'SMTP требует SSL или STARTTLS')
        try:config['port']=int(config.get('port',465 if config['security']=='ssl' else 587))
        except (ValueError,TypeError):raise HTTPException(503,'Некорректный порт SMTP') from None
        if not 1<=config['port']<=65535:raise HTTPException(503,'Некорректный порт SMTP')
        config['from_address']=email_address(config['from_address'])
        if config.get('username') and not config.get('password'):raise HTTPException(503,'Не указан пароль SMTP в локальной настройке')
    return Delivery(config,public_url,mode)


def public_status():
    try:
        delivery=prepare()
        return {'ready':True,'mode':delivery.mode,'public_url':delivery.public_url,'lifetime_s':60}
    except HTTPException as exc:return {'ready':False,'mode':'unavailable','message':exc.detail,'lifetime_s':60}
