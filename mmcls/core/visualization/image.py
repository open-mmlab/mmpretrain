import matplotlib.pyplot as plt
import mmcv
import numpy as np
from matplotlib.backend_bases import CloseEvent
from matplotlib.blocking_input import BlockingInput

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


class BaseImshowContextManager:
    """Context Manager to reuse matplotlib figure.

    Args:
        fig_size (tuple[int]): Size of the figure to show image.
    """

    def __init__(self, fig_size=(15, 10)) -> None:
        # Because save and show need different figure size
        # We set two figure and axes to handle save and show
        self.fig_save: plt.Figure = None
        self.ax_save: plt.Axes = None
        self.fig_show: plt.Figure = None
        self.ax_show: plt.Axes = None
        self.blocking_input: BlockingInput = None

        self.fig_size = fig_size

    def __enter__(self):
        self._initialize_fig_save()
        self._initialize_fig_show()
        return self

    def _initialize_fig_save(self):
        # A proper dpi for image save with default font size.
        fig = plt.figure(frameon=False, dpi=36)
        ax = fig.add_subplot()
        ax.axis('off')

        # remove white edges by set subplot margin
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

        self.fig_save, self.ax_save = fig, ax

    def _initialize_fig_show(self):
        # fig_save will be resized to image size, only fig_show needs fig_size.
        fig = plt.figure(frameon=False, figsize=self.fig_size)
        ax = fig.add_subplot()
        ax.axis('off')

        # remove white edges by set subplot margin
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

        self.fig_show, self.ax_show = fig, ax
        self.blocking_input = BlockingInput(
            self.fig_show,
            eventslist=('button_press_event', 'key_press_event',
                        'close_event'))

    def __exit__(self, exc_type, exc_value, traceback):
        plt.close(self.fig_save)
        plt.close(self.fig_show)

    def prepare(self):
        # If users force to destroy the window, rebuild fig_show.
        if not plt.fignum_exists(self.fig_show.number):
            self._initialize_fig_show()

        # Clear all axes
        self.ax_save.cla()
        self.ax_save.axis('off')
        self.ax_show.cla()
        self.ax_show.axis('off')

    def wait_continue(self, timeout=0):
        while True:
            # Disable matplotlib default hotkey to close figure.
            with plt.rc_context({'keymap.quit': []}):
                key_press = self.blocking_input(n=1, timeout=timeout)

            # Timeout or figure is closed or press space or press 'q'
            if len(key_press) == 0 or isinstance(
                    key_press[0],
                    CloseEvent) or key_press[0].key in ['q', ' ']:
                break


class ImshowInfosContextManager(BaseImshowContextManager):
    """Context Manager to reuse matplotlib figure and put infos on images."""

    def _put_text(self, ax, text, x, y, text_color, font_size):
        ax.text(
            x,
            y,
            f'{text}',
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

    def put_img_infos(self,
                      img,
                      infos,
                      text_color='white',
                      font_size=26,
                      row_width=20,
                      win_name='',
                      show=True,
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
            wait_time (int): How many seconds to display the image.
                Defaults to 0.
            out_file (Optional[str]): The filename to write the image.
                Defaults to None.

        Returns:
            np.ndarray: The image with extra infomations.
        """
        self.prepare()

        text_color = color_val_matplotlib(text_color)
        img = mmcv.imread(img).astype(np.uint8)

        x, y = 3, row_width // 2
        img = mmcv.bgr2rgb(img)
        width, height = img.shape[1], img.shape[0]
        img = np.ascontiguousarray(img)

        # add a small EPS to avoid precision lost due to matplotlib's
        # truncation (https://github.com/matplotlib/matplotlib/issues/15363)
        dpi = self.fig_save.get_dpi()
        self.fig_save.set_size_inches((width + EPS) / dpi,
                                      (height + EPS) / dpi)

        for k, v in infos.items():
            if isinstance(v, float):
                v = f'{v:.2f}'
            label_text = f'{k}: {v}'
            self._put_text(self.ax_save, label_text, x, y, text_color,
                           font_size)
            if show:
                self._put_text(self.ax_show, label_text, x, y, text_color,
                               font_size)
            y += row_width

        self.ax_save.imshow(img)
        stream, _ = self.fig_save.canvas.print_to_buffer()
        buffer = np.frombuffer(stream, dtype='uint8')
        img_rgba = buffer.reshape(height, width, 4)
        rgb, _ = np.split(img_rgba, [3], axis=2)
        img_save = rgb.astype('uint8')
        img_save = mmcv.rgb2bgr(img_save)

        if out_file is not None:
            mmcv.imwrite(img_save, out_file)

        if show:
            # Reserve some space for the tip.
            self.ax_show.set_title(win_name)
            self.ax_show.set_ylim(height + 20)
            self.ax_show.text(
                width // 2,
                height + 18,
                'Press SPACE to continue.',
                ha='center',
                fontsize=font_size)
            self.ax_show.imshow(img)

            # Refresh canvas, necessary for Qt5 backend.
            self.fig_show.canvas.draw()

            self.wait_continue(timeout=wait_time)

        return img_save


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
