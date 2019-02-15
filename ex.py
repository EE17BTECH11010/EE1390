import numpy as np
import matplotlib.pyplot as plt
import math

A=np.array([2,3])
B=np.array([3,-1])
p=np.array([-1,4])
N=np.vstack((A,B))
I=np.matmul(np.linalg.inv(N),p)
#print(I)
C=np.array([-0.5,0])
omat=np.array([[0,1],[-1,0]])
def dir_vec(X):
	return np.matmul(np.linalg.inv(omat),X)
#print(dir_vec(A))
#print(dir_vec(B))
p=dir_vec(A)
q=dir_vec(B)
len=100
y=np.linspace(-2,2,len)
z=np.linspace(-2,2,len)
x_AB=np.zeros((2,len))
x_IA=np.zeros((2,len))
for i in range(len):
	temp1=I+y[i]*dir_vec(A)
	x_AB[:,i]=temp1.T
	temp2=I+z[i]*dir_vec(B)
	x_IA[:,i]=temp2.T

	


plt.plot(x_AB[0,:],x_AB[1,:],label='$AC$',color = "red" )
plt.plot(x_IA[0,:],x_IA[1,:],label='$BD$',color = "red")


plt.plot(I[0],I[1],'o',color="green")
plt.text(I[0]*1.1,I[1]*1.1,'O')
R=5

t=np.linspace(0,2*np.pi,100)
x=R*np.cos(t)+I[0]
y=R*np.sin(t)+I[1]

k=math.atan(-2/3.0)
#print(R*np.cos(k)+I[0])
#print(R*np.sin(k)+I[1])


M=np.array([R*np.cos(k)+I[0],R*np.sin(k)+I[1]])

N=2*I-M

j=math.atan(3/1.0)
L=np.array([R*np.cos(j)+I[0],R*np.sin(j)+I[1]])
H=2*I-L

#print(M)
#print(N)
#print(L)
#print(H)



y1=np.linspace(-2,2,len)
y2=np.linspace(-2,2,len)
x1_AB=np.zeros((2,len))
x1_IA=np.zeros((2,len))
for i in range(len):
	temp3=M+y1[i]*dir_vec(N-I)
	x1_AB[:,i]=temp3.T
	temp4=N+y2[i]*dir_vec(N-I)
	x1_IA[:,i]=temp4.T
	

plt.plot(x1_AB[0,:],x1_AB[1,:],label='$PQ$',color="blue")
plt.plot(x1_IA[0,:],x1_IA[1,:],label='$QR$',color="blue")

y3=np.linspace(-1.5,1.5,len)
y4=np.linspace(-1.5,1.5,len)
x2_AB=np.zeros((2,len))
x2_IA=np.zeros((2,len))
for i in range(len):
	temp5=L+y3[i]*dir_vec(L-I)
	x2_AB[:,i]=temp5.T
	temp6=H+y4[i]*dir_vec(L-I)
	x2_IA[:,i]=temp6.T	


plt.plot(x2_AB[0,:],x2_AB[1,:],label='$RS$',color="blue")
plt.plot(x2_IA[0,:],x2_IA[1,:],label='$PS$',color="blue")

P=(L+M)/2	
pl=np.linalg.norm(P-L)

#print(pl)

pi=np.linalg.norm(P-I)
#print(pi)

h=(pl/pi)*5
#print(h)
k=np.sqrt((h*h)-(pl*pl))

Q=(k+pi)*P-k*I
Q=Q/pi
P=2*I-Q
#print(Q)
#print(P)


P1=(N+L)/2
rl=np.linalg.norm(P1-N)

#print(rl)

ri=np.linalg.norm(P1-I)
#print(ri)

h1=(rl/ri)*5
#print(h1)
k1=np.sqrt((h1*h1)-(rl*rl))

R=(k1+ri)*P1-k1*I
R=R/ri
S=2*I-R
#print(R)
#print(S)


e=np.linalg.norm(R-Q)
f=np.linalg.norm(R-P)
g=np.linalg.norm(P-Q)
s=(e+f+g)/2
area=np.sqrt(s*(s-f)*(s-g)*(s-e))
area_q=2*area
print('the cetre of triangle is ')
print(I)
print('the area of quadriletaral formed is')
print(area_q)

plt.plot(M[0],M[1],'o',color="green")
plt.text(M[0]*1.1,M[1]*1.1,'A')
plt.plot(N[0],N[1],'o',color="green")
plt.text(N[0]*1.1,N[1]*1.1,'C')
plt.plot(L[0],L[1],'o',color="green")
plt.text(L[0]*1.1,L[1]*1.1,'B')
plt.plot(H[0],H[1],'o',color="green")
plt.text(H[0]*1.1,H[1]*1.1,'D')
plt.plot(Q[0],Q[1],'o',color="green")
plt.text(Q[0]*1.1,Q[1]*1.1,'Q')
plt.plot(P[0],P[1],'o',color="green")
plt.text(P[0]*1.2,P[1]*1.2,'S')
plt.plot(R[0],R[1],'o',color="green")
plt.text(R[0]*1.1,R[1]*1.1,'P')
plt.plot(S[0],S[1],'o',color="green")
plt.text(S[0]*1.2,S[1]*1.2,'R')
plt.axis('equal')
plt.plot(x,y)
plt.xlabel('$x$')
plt.xlabel('$y$')
plt.legend(loc='best')
plt.grid()
plt.show()
plt.show()
