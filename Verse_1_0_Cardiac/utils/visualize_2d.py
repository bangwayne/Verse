import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import os
import torch
import monai.transforms as transforms



def draw_attention_masks(image, attn_mask, slice_name, save_dir):
    # Resize attn_mask to 384x384
    attn_mask_resized = F.interpolate(attn_mask.unsqueeze(1), size=(256, 256), mode='bilinear',
                                      align_corners=False).squeeze(1)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    for q_index in range(attn_mask_resized.shape[0]):
        mask = attn_mask_resized[i].cpu().numpy()
        img_2d = image[:, :, :].detach().cpu().numpy()

        # if np.sum(label_2d) == 0 or np.sum(preds_2d) == 0:
        #     continue
        array_min = img_2d.min()
        array_max = img_2d.max()
        normalized_image = (img_2d - array_min) / (array_max - array_min + 0.0001)

        # Scale to [0, 224]
        scaled_array = (normalized_image * 255).clip(0, 255).astype(np.uint8)
        #
        # img_2d = img_2d * 255
        scaled_img = scaled_array.transpose((1, 2, 0))
        # orginal img
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(scaled_img)
        ax1.set_title('Image')
        ax1.axis('off')

        # gt
        ax2.imshow(scaled_img, cmap='gray')
        show_mask(mask, ax2)
        ax2.set_title('Ground truth')
        ax2.axis('off')

        fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
        plt.savefig(os.path.join(save_dir, f'{q_index}_.png'), bbox_inches='tight')
        plt.close()


def draw_pred_result(image, preds, gt2D, work_dir, catalog, slice_name):

    preds = (preds > 0.5).int()

    root_dir = os.path.join(work_dir, slice_name)

    # print(gt2D.shape)
    if not os.path.exists(root_dir):
        os.makedirs(root_dir, exist_ok=True)

    # root_dir = os.path.join(root_dir,slice_name)
    # if not os.path.exists(root_dir):
    #     os.makedirs(root_dir)


    img_2d = image[:,:,:].detach().cpu().numpy()
    preds_2d = preds[:, :].detach().cpu().numpy()
    label_2d = gt2D[:, :].detach().cpu().numpy()
    # if np.sum(label_2d) == 0 or np.sum(preds_2d) == 0:
    #     continue
    array_min = img_2d.min()
    array_max = img_2d.max()
    normalized_image = (img_2d - array_min) / (array_max - array_min + 0.0001)

    # Scale to [0, 224]
    scaled_array = (normalized_image * 255).clip(0, 255).astype(np.uint8)
    #
    # img_2d = img_2d * 255
    scaled_img = scaled_array.transpose((1, 2, 0))
    # orginal img
    fig, (ax2, ax3) = plt.subplots(1, 2)

    # gt
    ax2.imshow(scaled_img, cmap='gray')
    show_mask(label_2d, ax2)
    ax2.set_title('Ground truth')
    ax2.axis('off')
    #
    # preds
    ax3.imshow(scaled_img, cmap='gray')
    show_mask(preds_2d, ax3)
    ax3.set_title('Prediction')
    ax3.axis('off')

    # boxe

    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
    plt.savefig(os.path.join(root_dir, f'{catalog}_.png'), bbox_inches='tight')
    plt.close()

def draw_result(image, preds, gt2D, work_dir, catalog, slice_name):

    preds = (preds > 0.5).int()

    root_dir = os.path.join(work_dir, slice_name)

    # print(gt2D.shape)
    if not os.path.exists(root_dir):
        os.makedirs(root_dir, exist_ok=True)

    # root_dir = os.path.join(root_dir,slice_name)
    # if not os.path.exists(root_dir):
    #     os.makedirs(root_dir)


    img_2d = image[:,:,:].detach().cpu().numpy()
    preds_2d = preds[:, :].detach().cpu().numpy()
    label_2d = gt2D[:, :].detach().cpu().numpy()
    # if np.sum(label_2d) == 0 or np.sum(preds_2d) == 0:
    #     continue
    array_min = img_2d.min()
    array_max = img_2d.max()
    normalized_image = (img_2d - array_min) / (array_max - array_min + 0.0001)

    # Scale to [0, 224]
    scaled_array = (normalized_image * 255).clip(0, 255).astype(np.uint8)
    #
    # img_2d = img_2d * 255
    scaled_img = scaled_array.transpose((1, 2, 0))
    # orginal img
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.imshow(scaled_img)
    ax1.set_title('Image')
    ax1.axis('off')

    # gt
    ax2.imshow(scaled_img, cmap='gray')
    show_mask(label_2d, ax2)
    ax2.set_title('Ground truth')
    ax2.axis('off')
    #
    # preds
    ax3.imshow(scaled_img, cmap='gray')
    show_mask(preds_2d, ax3)
    ax3.set_title('Prediction')
    ax3.axis('off')

    # boxe

    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
    plt.savefig(os.path.join(root_dir, f'{catalog}_.png'), bbox_inches='tight')
    plt.close()

def draw_result_with_point(image, preds, gt2D, work_dir, catalog, slice_name, point_tuple, iter_num):

    preds = (preds > 0.5).int()

    root_dir = os.path.join(work_dir, slice_name)

    # print(gt2D.shape)
    if not os.path.exists(root_dir):
        os.makedirs(root_dir, exist_ok=True)

    # root_dir = os.path.join(root_dir,slice_name)
    # if not os.path.exists(root_dir):
    #     os.makedirs(root_dir)


    img_2d = image[:,:,:].detach().cpu().numpy()
    preds_2d = preds[:, :].detach().cpu().numpy()
    label_2d = gt2D[:, :].detach().cpu().numpy()
    # if np.sum(label_2d) == 0 or np.sum(preds_2d) == 0:
    #     continue
    array_min = img_2d.min()
    array_max = img_2d.max()
    normalized_image = (img_2d - array_min) / (array_max - array_min + 0.0001)

    # Scale to [0, 224]
    scaled_array = (normalized_image * 255).clip(0, 255).astype(np.uint8)
    #
    # img_2d = img_2d * 255
    scaled_img = scaled_array.transpose((1, 2, 0))
    # orginal img
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.imshow(scaled_img)
    ax1.set_title('Image')
    ax1.axis('off')

    # gt
    ax2.imshow(scaled_img, cmap='gray')
    show_mask(label_2d, ax2)
    ax2.set_title('Ground truth')
    ax2.axis('off')
    #
    # preds
    ax3.imshow(scaled_img, cmap='gray')
    show_mask(preds_2d, ax3)
    for i in range(len(point_tuple[0])):
        point_coords = point_tuple[0][i].numpy()  # Assuming point_tuple['point'] is a tensor
        point_label = point_tuple[1][i].item()
        # print(point_coords)
        # Assuming point_tuple['labels'] is a tensor
        if point_label>-1:
            show_points(point_coords, point_label, ax3)
    ax3.set_title('Prediction')
    ax3.axis('off')

    # for i in range(len(point_tuple)):
    #     point_coords = point_tuple[0][i].numpy()  # Assuming point_tuple['point'] is a tensor
    #     point_label = point_tuple[1][i].item()
    #     # Assuming point_tuple['labels'] is a tensor
    #     if point_label>-1:
    #         show_points(point_coords, point_label, ax3)

    # boxe

    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
    plt.savefig(os.path.join(root_dir, f'{catalog}_iter{iter_num}_.png'), bbox_inches='tight')
    plt.close()


# def draw_result(category, image, bboxes, points, logits, gt2D, spatial_size, work_dir,slice_name):
#     zoom_out_transform = transforms.Compose([
#         transforms.AddChanneld(keys=["image", "label", "logits"]),
#         transforms.Resized(keys=["image", "label", "logits"], spatial_size=spatial_size, mode='nearest-exact')
#     ])
#     post_item = zoom_out_transform({
#         'image': image,
#         'label': gt2D,
#         'logits': logits
#     })
#     image, gt2D, logits = post_item['image'][0], post_item['label'][0], post_item['logits'][0]
#     preds = torch.sigmoid(logits)
#     preds = (preds > 0.5).int()
#
#     root_dir = work_dir + "/text_and_point_fig_examples_with3D/" + slice_name[0:10]
#
#     # print(gt2D.shape)
#     if not os.path.exists(root_dir):
#         os.makedirs(root_dir)
#
#     root_dir = os.path.join(root_dir,slice_name)
#     if not os.path.exists(root_dir):
#         os.makedirs(root_dir)
#
#     np.save(os.path.join(root_dir, category[0:4] + "_preds.npy"), preds.cpu().numpy())
#     np.save(os.path.join(root_dir, category[0:4] + "_gt.npy"), gt2D.cpu().numpy())
#
#
#     if bboxes is not None:
#         x1, y1, x2, y2 = bboxes[0].cpu().numpy()
#     if points is not None:
#         points = (points[0].cpu().numpy(), points[1].cpu().numpy())
#         points_ax = points[0][0]   # [n, 3]
#         points_label = points[1][0] # [n]
#
#
#     img_2d = image[:, :].detach().cpu().numpy()
#     preds_2d = preds[:, :].detach().cpu().numpy()
#     label_2d = gt2D[:, :].detach().cpu().numpy()
#     # if np.sum(label_2d) == 0 or np.sum(preds_2d) == 0:
#     #     continue
#
#     img_2d = img_2d * 255
#     # orginal img
#     fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
#     ax1.imshow(img_2d, cmap='gray')
#     ax1.set_title('Image')
#     ax1.axis('off')
#
#     # gt
#     ax2.imshow(img_2d, cmap='gray')
#     show_mask(label_2d, ax2)
#     ax2.set_title('Ground truth')
#     ax2.axis('off')
#
#     # preds
#     ax3.imshow(img_2d, cmap='gray')
#     show_mask(preds_2d, ax3)
#     ax3.set_title('Prediction')
#     ax3.axis('off')
#
#     # boxes
#     if bboxes is not None:
#         # if j >= x1 and j <= x2:
#         show_box((y1, x1, y2, x2), ax1)
#     # points
#     if points is not None:
#         for point_idx in range(points_label.shape[0]):
#             point = points_ax[point_idx]
#             label = points_label[point_idx] # [1]
#             # if j == point[0]:
#             show_points(point, label, ax1)
#
#     fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
#     plt.savefig(os.path.join(root_dir, f'{category}_.png'), bbox_inches='tight')
#     plt.close()

def show_mask(mask, ax):
    # color = np.array([251/255, 252/255, 0.6])
    # color_array = np.array([[0.5*255, 0.2*255, 0.2*255],
    #                         [0.2 * 255, 0.5 * 255, 0.2 * 255]
    #                        ])
    # h, w = mask.shape[-2:]
    # mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    colors = np.array([
        [0, 0, 0],       # Background (black)
        [255*0.8, 255*0.2, 255*0.2],     # Label 1 (red)
        [255*0.2, 255*0.8, 255*0.2],     # Label 2 (green)
        [255*0.2, 255*0.2, 255*0.8],     # Label 3 (blue)
        [255*0.8, 255*0.8, 255*0.2],   # Label 4 (yellow)
        # Add more colors if you have more labels
    ])

    # Map each label to its corresponding color
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for label in range(colors.shape[0]):
        color_mask[mask == label] = colors[label]

    # Display the color-masked image
    # ax.imshow(color_mask, alpha=0.6)
    array_min = color_mask.min()
    array_max = color_mask.max()
    normalized_image = (color_mask - array_min) / (array_max - array_min + 0.0001)

    # Scale to [0, 224]
    mask_image = (normalized_image * 255).clip(0, 255).astype(np.uint8)
    # mask_image = np.clip(mask_image, 0, 255).astype(np.uint8)
    ax.imshow(mask_image, alpha=0.6)

    # array_min = mask_image.min()
    # array_max = mask_image.max()
    # normalized_image = (mask_image - array_min) / (array_max - array_min + 0.0001)
    #
    # # Scale to [0, 224]
    # mask_image = (normalized_image * 255).clip(0, 255).astype(np.uint8)
    # # mask_image = np.clip(mask_image, 0, 255).astype(np.uint8)
    # ax.imshow(mask_image, alpha=0.6)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='blue', facecolor=(0,0,0,0), lw=2))

def show_points(points_ax, points_label, ax):
    color = 'yellow' if points_label == 0 else 'blue'
    ax.scatter(points_ax[1], points_ax[0], c=color, marker='o', s=3)