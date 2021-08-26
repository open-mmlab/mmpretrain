import matplotlib.pyplot as plt
import mmcv
import numpy as np

# A small value
EPS = 1e-2


def color_val_matplotlib(color):
    """Convert various input in BGR order to normalized RGB matplotlib color
    tuples,

    Args:
        color (:obj:`mmcv.Color`/str/tuple/int/ndarray): Color inputs

    Returns:
        tuple[float]: A tuple of 3 normalized floats indicating RGB channels.
    """
    color = mmcv.color_val(color)
    color = [color / 255 for color in color[::-1]]
    return tuple(color)


def imshow_infos(img,
                 infos,
                 text_color='white',
                 font_size=26,
                 row_width=20,
                 win_name='',
                 show=True,
                 fig_size=(15, 10),
                 wait_time=0,
                 out_file=None):
    """Show image with extra infomation.

    Args:
        img (str | ndarray): The image to be displayed.
        infos (dict): Extra infos to display in the image.
        text_color (:obj:`mmcv.Color`/str/tuple/int/ndarray): Extra infos
            display color. Defaults to 'white'.
        font_size (int): Extra infos display font size. Defaults to 26.
        row_width (int): width between each row of results on the image.
        win_name (str): The image title. Defaults to ''
        show (bool): Whether to show the image. Defaults to True.
        fig_size (tuple): Image show figure size. Defaults to (15, 10).
        wait_time (int): How many seconds to display the image. Defaults to 0.
        out_file (Optional[str]): The filename to write the image.
            Defaults to None.

    Returns:
        np.ndarray: The image with extra infomations.
    """
    img = mmcv.imread(img).astype(np.uint8)

    x, y = 3, row_width // 2
    text_color = color_val_matplotlib(text_color)

    img = mmcv.bgr2rgb(img)
    width, height = img.shape[1], img.shape[0]
    img = np.ascontiguousarray(img)

    # A proper dpi for image save with default font size.
    fig = plt.figure(win_name, frameon=False, figsize=fig_size, dpi=36)
    plt.title(win_name)
    canvas = fig.canvas
    dpi = fig.get_dpi()
    # add a small EPS to avoid precision lost due to matplotlib's truncation
    # (https://github.com/matplotlib/matplotlib/issues/15363)
    fig.set_size_inches((width + EPS) / dpi, (height + EPS) / dpi)

    # remove white edges by set subplot margin
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax = plt.gca()
    ax.axis('off')

    for k, v in infos.items():
        if isinstance(v, float):
            v = f'{v:.2f}'
        label_text = f'{k}: {v}'
        ax.text(
            x,
            y,
            f'{label_text}',
            bbox={
                'facecolor': 'black',
                'alpha': 0.7,
                'pad': 0.2,
                'edgecolor': 'none',
                'boxstyle': 'round'
            },
            color=text_color,
            fontsize=font_size,
            family='monospace',
            verticalalignment='top',
            horizontalalignment='left')
        y += row_width

    plt.imshow(img)
    stream, _ = canvas.print_to_buffer()
    buffer = np.frombuffer(stream, dtype='uint8')
    img_rgba = buffer.reshape(height, width, 4)
    rgb, _ = np.split(img_rgba, [3], axis=2)
    img = rgb.astype('uint8')
    img = mmcv.rgb2bgr(img)

    if show:
        # Matplotlib will adjust text size depends on window size and image
        # aspect ratio. It's hard to get, so here we set an adaptive dpi
        # according to screen height. 20 here is an empirical parameter.
        fig_manager = plt.get_current_fig_manager()
        if hasattr(fig_manager, 'window'):
            # Figure manager doesn't have window if no screen.
            screen_dpi = fig_manager.window.winfo_screenheight() // 20
            fig.set_dpi(screen_dpi)

        # We do not use cv2 for display because in some cases, opencv will
        # conflict with Qt, it will output a warning: Current thread
        # is not the object's thread. You can refer to
        # https://github.com/opencv/opencv-python/issues/46 for details
        if wait_time == 0:
            plt.show()
        else:
            plt.show(block=False)
            plt.pause(wait_time)
    if out_file is not None:
        mmcv.imwrite(img, out_file)

    plt.close()

    return img
