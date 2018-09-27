from smtplib import SMTP_SSL
import mimetypes
from email.message import EmailMessage
import os

class EmailUtil:
    def __init__(self,From,password,To,subject,content,attachments):
        self.msg = EmailMessage()
        self.msg["From"] = From
        self.password = password
        self.msg["To"] = To
        self.msg["Subject"] = subject
        self.content = content
        self.attachments = attachments
# msg = EmailMessage()

    def init(self):
        # self.msg.attach(self.content)
        # self.msg.preamble = 'You will not see this in a MIME-aware mail reader.\n'
        self.msg.set_content(self.content)
        self.server = SMTP_SSL("smtp.exmail.qq.com",465)
        self.server.login(self.msg["From"],self.password)


    def getMainType(self,file):
        return mimetypes.guess_type(file)[0].split("/",1)

    def add_attachement(self):
        for attachment in self.attachments:
            maintype,subtype = self.getMainType(attachment)
            with open(attachment,"rb") as f:
                self.msg.add_attachment(f.read(),
                                        maintype=maintype,
                                        subtype=subtype,
                                        filename=attachment.split(os.sep)[-1]
                                        )

    def send(self):
        self.init()
        self.add_attachement()
        self.server.send_message(self.msg)
        self.server.quit()

if __name__ == "__main__":
    # print(mimetypes.guess_type(r"E:\PWORKSPACE\testUwsgi\test.xlsx"))
    From = "dingxian@5i5j.com"
    passwd = ""
    To = "dingxian@5i5j.com"
    subject = "test email"
    content = "this is a test email"
    attachments = [r"E:\PWORKSPACE\testUwsgi\test.xlsx"]
    eml = EmailUtil(From,passwd,To,subject,content,attachments)
    eml.send()