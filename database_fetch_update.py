import check
import pymysql as sql

con=sql.connect("localhost","DuanLuYun","1234","GRADES")

cursor=con.cursor()

sql1="SELECT question_path FROM Student"

i=0

try:
    cursor.execute(sql1)
    result=cursor.fetchall()
    for row in result:
        answer=check.imagecheck(row[0])
        sql2="UPDATE Student SET answer=%d WHERE question_id=%i"%(answer,i)
        try:
            cursor.execute(sql2)
            con.commit()
        except:
            con.rollback()
        i+=1

except:
    print("unable to fetch data")

con.close()
