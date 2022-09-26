from jmetal.core.problem import Problem
from jmetal.util.observer import Observer
from AnalysisToolkit_LSH.MOQMKP_Package.dataset import Dataset


class BasicConfig:
    dataset_filename = "trainDataset.txt"
    detail_information_filename = "detailInfo.txt"
    compare_result_filename = "compare_result.txt"
    plotting_dataset_filename = "plot_dataset.txt"

    @staticmethod
    def get_filename_lis():
        tmp = BasicConfig
        return [tmp.dataset_filename, tmp.detail_information_filename,
                tmp.compare_result_filename, tmp.plotting_dataset_filename]


class ShowDetailInformation(Observer):
    def __init__(self, problem: Problem):
        self.problem = problem
        self.receive_data: Dataset = None               # （在updata中更新）

        self.clear_file()                               # 清空文件内容，避免手动删除

    def clear_file(self):
        filename_list = BasicConfig.get_filename_lis()
        for item in filename_list:
            with open(item, 'w+', encoding='utf-8') as f:
                pass

    def update(self, *args, **kwargs):
        self.receive_data = kwargs["dataset"]
        self.save_dataset_to_train_network()
        self.save_detail_information()
        self.save_different_result_dataset()
        self.save_plotting_dataset()

    def save_dataset_to_train_network(self):
        dataset = self.receive_data.datasetToNNetwork
        dataset_filename = BasicConfig.dataset_filename
        with open(dataset_filename, 'a+', encoding='utf-8') as f:
            for target, associated in dataset:
                if len(target) != 0 and len(associated) != 0:
                    for item in associated:
                        f.write(f"{target}:{item}\n")

    def save_detail_information(self):
        dataset = self.receive_data.datasetToNNetwork
        detail_information_filename = BasicConfig.detail_information_filename
        generations = self.receive_data.current_generations
        reference_points = self.receive_data.reference_points
        number_of_target = len(self.receive_data.target)
        number_of_associated = len(self.receive_data.associated)
        with open(detail_information_filename, 'a+', encoding='utf-8') as f:
            f.write(f"第{generations}代, (从1开始计数)\n")
            f.write(f"target共{number_of_target}个\n")
            f.write(f"associated共{number_of_associated}个\n")
            for reference_index in range(len(reference_points)):
                f.write(f"{reference_index}\t{reference_points[reference_index]}\t{dataset[reference_index][0]}\t")
                f.write(f"{dataset[reference_index][1]}\n")
            f.write('\n---------------------------------------------------------------------------------------------\n')

    def save_different_result_dataset(self):
        with_compare_result_dataset = self.receive_data.with_compare_result_datasetToNNetwork
        compare_result_filename = BasicConfig.compare_result_filename
        generations = self.receive_data.current_generations
        amount = [0, 0, 0]
        with open(compare_result_filename, 'a+', encoding='utf-8') as f:
            f.write(f"第{generations}代\n")
            for target, associated, result in with_compare_result_dataset:
                if result == -1:
                    amount[0] += 1
                elif result == 0:
                    amount[1] += 1
                elif result == 1:
                    amount[2] += 1
            # if result != 0:
            # 	f.write(f"{target}:{associated}\t{result}\n")
            f.write(f"-1的个数：{amount[0]}个\t0的个数：{amount[1]}个\t1的个数：{amount[2]}个\n")
            f.write('\n---------------------------------------------------------------------------------------------\n')

    def save_plotting_dataset(self):
        plotting_filename = BasicConfig.plotting_dataset_filename
        dataset = self.receive_data.plotting_objectives_dataset
        with open(plotting_filename, 'a+', encoding='utf-8') as f:
            f.write(f"{dataset}\n")


