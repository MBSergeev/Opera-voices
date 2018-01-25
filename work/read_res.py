fp1=open("out.dat")

data=fp1.read().splitlines()
data1=[]
for rec in data:
   pos1=rec.find("[")
   pos2=rec.find("]")
   arg =  rec[pos1+1:pos2].split()
   val=float(rec[pos2+1:])
   rec1=[]
   rec1.append(arg)
   rec1.append(val)
   data1.append(rec1)

fp1.close()

data2 = sorted(data1,key=lambda x: -x[1])[0:100]
for i in range(len(data2)):
   print data2[i]
