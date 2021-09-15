import wx, numpy as np
from threading import Thread
from sciwx.canvas import MCanvas
from sciwx.widgets import ParaDialog
import os.path as osp

from .camera import RandomCamera, USBCamera

devices = {'Random Noise': RandomCamera, 'USB Camera': USBCamera}

class Plugin ( wx.Panel ):
    title = 'Camera Panel'
    def __init__(self, parent, app=None):
        wx.Panel.__init__( self, parent, id = wx.ID_ANY, pos = wx.DefaultPosition, size = wx.Size(-1, -1), style = wx.TAB_TRAVERSAL)

        self.app = app

        bSizer1 = wx.BoxSizer( wx.VERTICAL )

        self.tool_bar = wx.ToolBar( self, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.TB_HORIZONTAL|wx.TB_HORZ_TEXT )
        self.m_staticText1 = wx.StaticText( self.tool_bar, wx.ID_ANY, u"Camera: ", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.m_staticText1.Wrap( -1 )

        self.tool_bar.AddControl( self.m_staticText1 )
        self.com_devices = wx.ComboBox( self.tool_bar, wx.ID_ANY, u"Combo!", wx.DefaultPosition, wx.DefaultSize, [], 0 )
        self.tool_bar.AddControl( self.com_devices )

        path = osp.abspath(osp.dirname(__file__))
        icon_png = osp.join(path, "next.png")

        self.tol_open = self.tool_bar.AddTool( wx.ID_ANY, u"Open", wx.Bitmap(icon_png), wx.NullBitmap, wx.ITEM_NORMAL, wx.EmptyString, wx.EmptyString, None )

        self.tol_start = self.tool_bar.AddTool( wx.ID_ANY, u"Start", wx.Bitmap(icon_png), wx.NullBitmap, wx.ITEM_NORMAL, wx.EmptyString, wx.EmptyString, None )

        self.tol_stop = self.tool_bar.AddTool( wx.ID_ANY, u"Stop", wx.Bitmap(icon_png), wx.NullBitmap, wx.ITEM_NORMAL, wx.EmptyString, wx.EmptyString, None )

        self.tol_close = self.tool_bar.AddTool( wx.ID_ANY, u"Close", wx.Bitmap(icon_png), wx.NullBitmap, wx.ITEM_NORMAL, wx.EmptyString, wx.EmptyString, None )

        self.tool_bar.AddSeparator()

        self.spn_cache = wx.SpinCtrl( self.tool_bar, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, wx.SP_ARROW_KEYS, 1, 1024, 0 )
        self.tool_bar.AddControl( self.spn_cache )

        self.btn_config = wx.Button( self.tool_bar, wx.ID_ANY, u"Config", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.tool_bar.AddControl( self.btn_config )
        self.tool_bar.Realize()

        bSizer1.Add( self.tool_bar, 0, wx.EXPAND, 5 )

        self.com_devices.SetItems(sorted(devices))
        self.com_devices.Select(0)

        self.SetSizer( bSizer1 )
        self.Layout()

        # Connect Events
        self.Bind( wx.EVT_TOOL, self.on_open, id = self.tol_open.GetId() )
        self.Bind( wx.EVT_TOOL, self.on_start, id = self.tol_start.GetId() )
        self.Bind( wx.EVT_TOOL, self.on_stop, id = self.tol_stop.GetId() )
        self.Bind( wx.EVT_TOOL, self.on_close, id = self.tol_close.GetId() )
        self.btn_config.Bind( wx.EVT_BUTTON, self.on_config )

        self.status = False
        self.camera = RandomCamera()
        
    def on_open(self, event):
        self.camera = devices[self.com_devices.GetValue()]()
        if not self.camera.view is None:
            dialog = ParaDialog(self, self.camera.name)
            view, para = self.camera.view, self.camera.para.copy()
            dialog.init_view(view, para, False)
            if dialog.show():
                self.camera.para.update(para)
                self.camera.open()
        else: self.camera.open()


        self.app.show_img([np.zeros((512, 512), np.uint8)], "Camera")

        self.app.alert("Ready to start.")

    def on_stop(self, event):
        self.status = False

    def on_close(self, event):
        self.camera.close()
        self.camera = RandomCamera()

    def on_start(self, event):
        self.status = True
        td = Thread(target=self.hold, args=())
        td.setDaemon(True)
        td.start()

    def on_config(self, event):
        dialog = ParaDialog(self, self.camera.name)
        view, para = self.camera.view, self.camera.para.copy()
        dialog.init_view(view, para, False)
        if dialog.show(): self.camera.para.update(para)
        
    def hold(self):
        while self.status:
            img = self.camera.frame()
            ips = self.app.get_img()
            if img.shape != ips.img.shape:
                ips.img = img
            else:
                cache = self.spn_cache.GetValue()
                ips.imgs = (ips.imgs + [img])[-cache:]
                ips.cur = len(ips.imgs) - 1
                wx.CallAfter(ips.update)
        
    def __del__( self ):
        pass

