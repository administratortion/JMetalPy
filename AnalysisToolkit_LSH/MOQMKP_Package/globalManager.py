from AnalysisToolkit_LSH.MOQMKP_Package.dataset import Dataset
from AnalysisToolkit_LSH.MOQMKP_Package.datasetInfo import ShowDetailInformation
from AnalysisToolkit_LSH.util.reference_points import UniformReferenceDirectionFactory
from AnalysisToolkit_LSH.MOQMKP_Package.plotting import Plotting
from jmetal.util.observer import Observer


class GlobalManager(Observer):
    def __init__(self, reference_points: UniformReferenceDirectionFactory):
        self.reference_pointers = reference_points                 # 参考点
        self.all_populations = []
        self.dataset = None
        self.problem = None

        self.is_had_dataset_instance = False            # 初始化为没有

    def update(self, *args, **kwargs):
        self.all_populations = kwargs["all_populations"]
        self.problem = kwargs["PROBLEM"]

        self.run(5)

    def run(self, interval):
        """

        Args:
            interval:  间隔内的所有种群数

        Returns:
                    然后 数据到文件中
        """
        if len(self.all_populations) >= interval:

            # 如果没有初始化dataset，就是初始化
            if self.is_had_dataset_instance is False:
                self.dataset = Dataset(self.reference_pointers, self.problem)       # 初始化
                self.dataset.register(ShowDetailInformation(self.problem))          # 注册 输出信息类
                # self.dataset.register(Plotting())                                 # 注册 绘图类
                self.is_had_dataset_instance = True

            self.dataset.add_current_population(self.all_populations)
            self.dataset.run_by_setting_interval(interval)
