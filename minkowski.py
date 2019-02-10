import pylab as pl
import base
import math
import matplotlib.animation as anim
import plot_opt

coq = [(666.5364448871007, 975.8245835899722), (593.2797318725675, 958.7881387028714), (548.9849751661054, 928.12253790609), (509.80115192577364, 882.1241367109178), (485.9501290838325, 817.3856461399348), (480.8391956177022, 740.7216441479812), (480.8391956177022, 674.2795090882881), (462.0991062418914, 647.0211972689269), (436.54443891124015, 641.9102638027966), (407.58248260316896, 660.6503531786076), (392.2496822047781, 720.2779102834603), (404.17519362574876, 790.1273343205736), (353.0658589644463, 756.054444546372), (320.6966136789549, 698.1305319302293), (320.6966136789549, 739.0179996592711), (274.69821248378275, 672.575864599578), (254.25447861926182, 599.3191515850447), (240.62532270958104, 635.0956858479564), (203.14514395795936, 578.8754177205237), (211.66336640150985, 483.4713263527593), (169.07225418375788, 491.9895487963097), (116.25927503374533, 471.54581493178875), (158.8503872514973, 415.32554680435624), (116.25927503374533, 403.40003538338556), (87.29731872567413, 369.3271456091841), (145.22123134181675, 328.4396778801422), (107.74105259019507, 311.40323299304146), (83.89002974825394, 273.92305424141955), (150.33216480794704, 255.1829648656087), (247.43990066442143, 241.55380895592816), (337.73305856605566, 255.1829648656087), (436.54443891124015, 294.3667881059405), (509.80115192577364, 347.17976725595304), (572.8359980080465, 420.4364802704863), (605.2052432935379, 503.91506021728026), (623.9453326693488, 566.9499062995532), (680.1656007967815, 553.3207503898726), (748.3113803451847, 519.247860615671), (809.6425819387475, 473.24945942049885), (847.1227606903692, 362.5125676543437), (886.306583930701, 307.99594401562126), (908.453962283932, 284.14492117368013), (874.3810725097305, 284.14492117368013), (870.9737835323103, 265.4048317978693), (886.306583930701, 244.96109793334836), (857.3446276226296, 210.88820815914664), (908.453962283932, 175.11167389623506), (910.1576067726421, 122.29869474622251), (961.2669414339446, 118.89140576880231), (1000.4507646742763, 122.29869474622251), (1063.485610756549, 76.30029355105034), (1111.1876564404313, 147.85336207687374), (1123.1131678614017, 202.36998571559639), (1102.669433996881, 239.85016446721806), (1112.8913009291416, 255.1829648656087), (1158.8897021243135, 261.9975428204491), (1174.2225025227042, 277.33034321883974), (1174.2225025227042, 287.5522101511003), (1138.4459682597928, 297.7740770833607), (1116.2985899065618, 309.69958850433136), (1140.1496127485027, 338.66154481240255), (1150.3714796807633, 372.73443458660427), (1141.853257237213, 398.28910191725527), (1118.0022343952717, 411.91825782693604), (1080.52205564365, 408.51096884951585), (1097.558500530751, 444.28750311242743), (1116.2985899065618, 503.91506021728026), (1119.705878883982, 575.4681287431035), (1099.2621450194608, 660.6503531786076), (1063.485610756549, 744.1289331254014), (1017.4872095613771, 810.5710681850945), (959.5632969452345, 875.3095587560775), (894.8248063742515, 924.7152489286698), (879.4920059758606, 950.269916259321), (850.5300496677894, 977.5282280786822), (796.0134260290667, 1013.3047623415939), (809.6425819387475, 1078.043252912577), (847.1227606903692, 1132.5598765512993), (893.1211618855414, 1163.2254773480809), (961.2669414339446, 1173.4473442803414), (1029.4127209823475, 1188.780144678732), (1049.8564548468687, 1199.0020116109924), (1063.485610756549, 1222.8530344529336), (1022.5981430275071, 1212.6311675206732), (993.636186719436, 1221.1493899642235), (956.1560079678143, 1207.520234054543), (881.1956504645707, 1195.5947226335725), (898.2320953516717, 1219.4457454755134), (872.6774280210204, 1222.8530344529336), (819.8644488710079, 1190.4837891674422), (777.2733366532559, 1204.1129450771227), (736.385868924214, 1204.1129450771227), (789.1988480742266, 1146.1890324609801), (780.6806256306761, 1135.9671655287195), (715.9421350596931, 1139.3744545061397), (661.4255114209707, 1100.190631265808), (659.7218669322606, 1069.5250304690264), (709.1275571048527, 1096.7833422883878), (750.0150248338946, 1105.3015647319382), (772.1624031871256, 1096.7833422883878), (736.385868924214, 1023.5266292738544), (700.6093346613025, 1009.8974733641737), (700.6093346613025, 1009.8974733641737), (683.5728897742017, 997.9719619432032), (666.5364448871007, 975.8245835899722)]
coq = [base.Vec2D(u, -v) for u, v in coq]
center = sum(coq, base.Vec2D(0, 0)) / len(coq)
star = []
for i in range(10):
	angle = math.pi * (0.5 + i / 5)
	r = 700 if i % 2 == 0 else 260
	star.append(center + base.Vec2D(math.cos(angle), math.sin(angle)) * r)
ps = coq + star
miX, miY, maX, maY = 10000, 10000, -10000, -10000
for p in ps:
	miX, maX = min(p.x, miX), max(p.x, maX) 
	miY, maY = min(p.y, miY), max(p.y, maY)
dx, dy = (maX - miX) / 300, (maY - miY) / 300

def mink(A, B, t, col):
	L = []
	for a in A:
		x, y = base.setToLists([t*a+(1-t)*b for b in B])
		L.append(pl.fill(x, y, color=col)[0])
	for b in B:
		x, y = base.setToLists([t*a+(1-t)*b for a in A])
		L.append(pl.fill(x, y, color=col)[0])
	return L

fig = pl.figure(figsize=(dx, dy))
ims = []
N = 45
for i in range(N):
	t = max(0, min(N-10, i-5)) / (N-10)
	ims.append(mink(star, coq, t, (t, 0, 1-t)))

ani = anim.ArtistAnimation(fig, ims, interval=90, repeat=True)
ani.save('res.gif', writer='imagemagick')
# pl.show()