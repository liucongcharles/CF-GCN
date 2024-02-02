
class DataConfig:
    data_name = ""
    root_dir = ""
    label_transform = "norm"
    def get_data_config(self, data_name):
        self.data_name = data_name
        if data_name == 'LEVIR':
            # self.root_dir = '/home/wxs/lc/LEVIR_normal'
            self.root_dir = '/Users/liucong/Desktop/研究生/论文/变化检测/LEVIR_normal'
        elif data_name == 'quick_start':
            self.root_dir = './samples/'
        elif data_name == 'WHU':
            self.root_dir = '/home/wxs/lc/WHU-CD-256'
        elif data_name == 'DSIFN':
            self.root_dir = '/home/wxs/lc/DSIFN-CD-256'
        else:
            raise TypeError('%s has not defined' % data_name)
        return self


if __name__ == '__main__':
    data = DataConfig().get_data_config(data_name='LEVIR')
    print(data.data_name)
    print(data.root_dir)
    print(data.label_transform)

