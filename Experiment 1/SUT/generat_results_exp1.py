import numpy as np, matplotlib.pyplot as plt
eps=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
data=np.array([
[0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,1,4,10,24],
[0,0,0,0,0,0,0,0,1,1],
[0,0,0,4,10,24,49,76,124,188],
[0,0,0,0,0,0,1,10,24,37],
[0,0,0,0,0,0,0,0,4,8],
[0,0,0,0,0,0,0,0,0,4],
[0,0,0,0,0,0,0,2,6,10],
[0,0,0,0,0,0,0,0,0,0]],dtype=int)

fig,ax=plt.subplots(figsize=(13,6))
im=ax.imshow(data,aspect='auto',origin='lower',cmap='YlOrRd',vmin=0,vmax=data.max())
ax.set_xticks(np.arange(len(eps))); ax.set_xticklabels(eps)
ax.set_yticks(np.arange(10)); ax.set_yticklabels(range(10))
ax.set_xlabel("Epsilon")
ax.set_ylabel("Digit Class")
ax.set_title("Adversarial Samples Across Classes and Epsilon-ball")
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        v=data[i,j]
        ax.text(j,i,str(v),ha='center',va='center',fontsize=8,color='black' if v<100 else 'white')
fig.colorbar(im,ax=ax,label="Number of Adversarial Samples")
plt.tight_layout(); 
plt.savefig("exp_1.pdf")
plt.savefig("exp_1.png")
plt.show()
