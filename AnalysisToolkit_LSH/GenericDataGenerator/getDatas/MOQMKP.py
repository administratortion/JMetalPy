from AnalysisToolkit_LSH.GenericDataGenerator.core.GenericDataGenerator\
    import MOQMKPGenerator, DensityEnum, FileCountEnum

generator = MOQMKPGenerator(
    number_of_variables=50,
    number_of_knapsacks=3,
    density=DensityEnum.Density_1_00,
    max_file_counts=FileCountEnum.File_Counts_1,
    cost_lb=1,
    cost_ub=10,
    union_cost_lb=1,
    union_cost_ub=10,
    item_wei_lb=50,
    item_wei_ub=100
).run()





