scores0=['c','c#','d','d#','e','f','f#','g','g#','a','a#','h']
freq=[]
for i in range(3,12*5+3):
        fr=55*(2)**(i/12.0)
        freq.append(fr)

scores=[]

for x in scores0:
        scores.append(x+'-mag')
for x in scores0:
        scores.append(x+'-min')
for x in scores0:
        scores.append(x+'-1')
for x in scores0:
        scores.append(x+'-2')
for x in scores0:
        scores.append(x+'-3')

for fr in zip(scores,freq):
        print fr

