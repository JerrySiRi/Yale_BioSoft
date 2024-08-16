import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv

load_dotenv()
password = os.getenv('Email_Password')

# 设置SMTP服务器的配置
smtp_server = 'smtp.gmail.com'
smtp_port = 587
sender_email = 'jerrysiri.xcdh@gmail.com'  # 你的 Gmail 地址
sender_password = password       # 必须开启2FA验证 + 使用应用密码
receiver_email = 'rui.shi@yale.edu'  # 收件人地址


# 创建发送邮件的函数
def send_email(subject, body):
    # 创建一个MIMEMultipart对象，用于包含邮件内容
    message = MIMEMultipart()
    message['From'] = sender_email
    message['To'] = receiver_email
    message['Subject'] = subject # 邮件主题

    # 添加邮件正文
    message.attach(MIMEText(body, 'plain')) # body是邮件内容

    # 连接到SMTP服务器并发送邮件
    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()  # 启用安全连接
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, receiver_email, message.as_string())
        print("Email sent successfully!")
    except Exception as e:
        print(f"Failed to send email: {e}")
    finally:
        server.quit()


# 示例：在程序运行过程中发送进度更新
def your_program():
    # 第一步
    step = "Step 1: Initializing data..."
    print(step)
    send_email("Program Progress Update", step)
    
    # 模拟程序运行
    import time
    time.sleep(1)  # 假设程序运行了2秒

    # 第二步
    step = "Step 2: Processing data..."
    print(step)
    send_email("Program Progress Update", step)
    
    time.sleep(1)

    # 第三步
    step = "Step 3: Finalizing..."
    print(step)
    send_email("Program Progress Update", step)
    
    time.sleep(1)

    # 完成
    step = "Program completed successfully."
    print(step)
    send_email("Program Progress Update", step)



if __name__ == "__main__":
    # 运行程序
    your_program()