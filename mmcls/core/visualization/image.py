# Copyright (c) OpenMMLab. All rights reserved.
import matplotlib.pyplot as plt
import mmcv
import numpy as np
from matplotlib.backend_bases import CloseEvent

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


class BaseFigureContextManager:
    """Context Manager to reuse matplotlib figure.

    It provides a figure for saving and a figure for showing to support
    different settings.

    Args:
        axis (bool): Whether to show the axis lines.
        fig_save_cfg (dict): Keyword parameters of figure for saving.
            Defaults to empty dict.
        fig_show_cfg (dict): Keyword parameters of figure for showing.
            Defaults to empty dict.
    """

    def __init__(self, axis=False, fig_save_cfg={}, fig_show_cfg={}) -> None:
        self.is_inline = 'inline' in plt.get_backend()

        # Because save and show need different figure size
        # We set two figure and axes to handle save and show
        self.fig_save: plt.Figure = None
        self.fig_save_cfg = fig_save_cfg
        self.ax_save: plt.Axes = None

        self.fig_show: plt.Figure = None
        self.fig_show_cfg = fig_show_cfg
        self.ax_show: plt.Axes = None

        self.axis = axis

    def __enter__(self):
        if not self.is_inline:
            # If use inline backend, we cannot control which figure to show,
            # so disable the interactive fig_show, and put the initialization
            # of fig_save to `prepare` function.
            self._initialize_fig_save()
            self._initialize_fig_show()
        return self

    def _initialize_fig_save(self):
        fig = plt.figure(**self.fig_save_cfg)
        ax = fig.add_subplot()

        # remove white edges by set subplot margin
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

        self.fig_save, self.ax_save = fig, ax

    def _initialize_fig_show(self):
        # fig_save will be resized to image size, only fig_show needs fig_size.
        fig = plt.figure(**self.fig_show_cfg)
        ax = fig.add_subplot()

        # remove white edges by set subplot margin
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

        self.fig_show, self.ax_show = fig, ax

    def __exit__(self, exc_type, exc_value, traceback):
        if self.is_inline:
            # If use inline backend, whether to close figure depends on if
            # users want to show the image.
            return

        plt.close(self.fig_save)
        plt.close(self.fig_show)

    def prepare(self):
        if self.is_inline:
            # if use inline backend, just rebuild the fig_save.
            self._initialize_fig_save()
            self.ax_save.cla()
            self.ax_save.axis(self.axis)
            return

        # If users force to destroy the window, rebuild fig_show.
        if not plt.fignum_exists(self.fig_show.number):
            self._initialize_fig_show()

        # Clear all axes
        self.ax_save.cla()
        self.ax_save.axis(self.axis)
        self.ax_show.cla()
        self.ax_show.axis(self.axis)

    def wait_continue(self, timeout=0, continue_key=' ') -> int:
        """Show the image and wait for the user's input.

        This implementation refers to
        https://github.com/matplotlib/matplotlib/blob/v3.5.x/lib/matplotlib/_blocking_input.py

        Args:
            timeout (int): If positive, continue after ``timeout`` seconds.
                Defaults to 0.
            continue_key (str): The key for users to continue. Defaults to
                the space key.

        Returns:
            int: If zero, means time out or the user pressed ``continue_key``,
                and if one, means the user closed the show figure.
        """  # noqa: E501
        if self.is_inline:
            # If use inline backend, interactive input and timeout is no use.
            return

        if self.fig_show.canvas.manager:
            # Ensure that the figure is shown
            self.fig_show.show()

        while True:

            # Connect the events to the handler function call.
            event = None

            def handler(ev):
                # Set external event variable
                nonlocal event
                # Qt backend may fire two events at the same time,
                # use a condition to avoid missing close event.
                event = ev if not isinstance(event, CloseEvent) else event
                self.fig_show.canvas.stop_event_loop()

            cids = [
                self.fig_show.canvas.mpl_connect(name, handler)
                for name in ('key_press_event', 'close_event')
            ]

            try:
                self.fig_show.canvas.start_event_loop(timeout)
            finally:  # Run even on exception like ctrl-c.
                # Disconnect the callbacks.
                for cid in cids:
                    self.fig_show.canvas.mpl_disconnect(cid)

            if isinstance(event, CloseEvent):
                return 1  # Quit for close.
            elif event is None or event.key == continue_key:
                return 0  # Quit for continue.


class ImshowInfosContextManager(BaseFigureContextManager):
    """Context Manager to reuse matplotlib figure and put infos on images.

    Args:
        fig_size (tuple[int]): Size of the figure to show image.

    Examples:
        >>> import mmcv
        >>> from mmcls.core import visualization as vis
        >>> img1 = mmcv.imread("./1.png")
        >>> info1 = {'class': 'cat', 'label': 0}
        >>> img2 = mmcv.imread("./2.png")
        >>> info2 = {'class': 'dog', 'label': 1}
        >>> with vis.ImshowInfosContextManager() as manager:
        ...     # Show img1
        ...     manager.put_img_infos(img1, info1)
        ...     # Show img2 on the same figure and save output image.
        ...     manager.put_img_infos(
        ...         img2, info2, out_file='./2_out.png')
    """

    def __init__(self, fig_size=(15, 10)):
        super().__init__(
            axis=False,
            # A proper dpi for image save with default font size.
            fig_save_cfg=dict(frameon=False, dpi=36),
            fig_show_cfg=dict(frameon=False, figsize=fig_size))

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
        """Show image with extra information.

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
            if show and not self.is_inline:
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

        ret = 0
        if show and not self.is_inline:
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

            ret = self.wait_continue(timeout=wait_time)
        elif (not show) and self.is_inline:
            # If use inline backend, we use fig_save to show the image
            # So we need to close it if users don't want to show.
            plt.close(self.fig_save)

        return ret, img_save


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
    """Show image with extra information.

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
    with ImshowInfosContextManager(fig_size=fig_size) as manager:
        _, img = manager.put_img_infos(
            img,
            infos,
            text_color=text_color,
            font_size=font_size,
            row_width=row_width,
            win_name=win_name,
            show=show,
            wait_time=wait_time,
            out_file=out_file)
    return img
