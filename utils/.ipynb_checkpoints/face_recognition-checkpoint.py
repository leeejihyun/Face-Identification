import insightface
import numpy as np
from numpy.linalg import norm

def identification(img, gallary_list, emb_list, model, threshold=0.65):
	sim_list=[]
	emb1 = model.get_embedding(img).flatten()
	for emb2 in emb_list:
		emb2 = emb2.flatten()
		sim2 = np.dot(emb1, emb2)/(norm(emb1)*norm(emb2))
		sim = 0.5 * (sim2 + 1)
		sim_list.append(sim)
	idx = sim_list.index(max(sim_list))
	sim_result = sim_list[idx]
	if sim_result > threshold:
		name_result = gallary_list[idx]
	else:
		name_result = 'nobody'

	return name_result