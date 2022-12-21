from pytorch3d.loss import chamfer_distance
import torch
import trimesh

one = torch.tensor([2, 0, 0]).cuda()
zero = torch.tensor([0, 0, 0]).cuda()
print(chamfer_distance(one[None, None].float(),
      zero[None, None].float(), norm=1))
exit()
# (tensor(2., device='cuda:0'), None)


mesh = trimesh.load("output/gt_mesh.obj")
points = mesh.sample(2**20)
sample_points = trimesh.Trimesh(vertices=points, faces=[])
sample_points.export("output/sampled_points.obj")
points_1 = torch.tensor(points).cuda()


mesh_2 = trimesh.load("output/x_hit.obj")
points_2 = torch.tensor(mesh_2.vertices).cuda()

print("points_1.shape")
print(points_1.shape)
print("points_2.shape")
print(points_2.shape)


bbox_diagonal = ((torch.min(points_1[..., 0]) + torch.max(points_1[..., 0]))**2 + (torch.min(points_1[..., 1]) +
                                                                                   torch.max(points_1[..., 1]))**2 + (torch.min(points_1[..., 2]) + torch.max(points_1[..., 2]))**2)**0.5

print("bbox_diagonal")
print(bbox_diagonal)

print(chamfer_distance(points_1[None], points_2[None], norm=1))
