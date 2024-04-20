import numpy as np
from scipy.optimize import minimize
from typing import List


class Pipeline:
    def __init__(self, pipe_length, pipe_area, height, name=""):
        self.name = name
        self.pipe_length = pipe_length
        self.pipe_area = pipe_area
        self.height = height

        self.drag_coefficient: float = 0.002
        self.flow_rate: float = 0

    def p_stat(self, fluid_density, gravity):
        return fluid_density * gravity * self.height

    def p_res(self, fluid_density):
        fluid_velocity = self.flow_rate / self.pipe_area
        return self.pipe_length * self.drag_coefficient * (fluid_density / 2) * fluid_velocity ** 2


class MultiPipelineModel:
    def __init__(self, name=""):
        self.p0: float = 1e6
        self.fluid_density: float = 1000
        self.gravity: float = 9.81
        self.power: float = 0

        self.pipelines: List[Pipeline] = []

        self.min_flow_rate: float = 0
        self.max_flow_rate: float = 2

        self.min_power = 200_000
        self.max_power = 1_000_000
        self.name = name

    def flow_rate_pump(self):
        return sum(pipeline.flow_rate for pipeline in self.pipelines)

    def p_pump(self):
        pump_flow_rate = self.flow_rate_pump()
        a = 4 / 27 * (self.p0 ** 3) / ((self.power - 1e5) ** 2)
        b = self.p0 / 2
        p_pump = self.p0 - a * pump_flow_rate ** 2 - b * pump_flow_rate ** 3
        return p_pump

    def differential_pressures(self):
        differential_pressures = []
        for pipeline in self.pipelines:
            dp = pipeline.p_stat(self.fluid_density, self.gravity) + pipeline.p_res(self.fluid_density) - self.p_pump()
            differential_pressures.append(dp)

        return differential_pressures

    def efficiency(self):
        efficiencies = []

        for pipeline in self.pipelines:
            efficiency = pipeline.p_stat(self.fluid_density, self.gravity) * pipeline.flow_rate / self.power
            efficiencies.append(efficiency)

        return np.sum(efficiencies)

    # solver functions
    def optimize_flow_rates(self, method="Nelder-Mead", **kwargs):
        xn = []

        def loss_function(x):
            xn.append(x)

            for i, pipeline in enumerate(self.pipelines):
                pipeline.flow_rate = x[i]

            # create error scalar value
            return np.sum(np.power(self.differential_pressures(), 2))

        # get start values
        x0 = np.array([pipeline.flow_rate for pipeline in self.pipelines])
        optimization = minimize(loss_function, x0, method=method, **kwargs)

        if not optimization.success:
            raise Exception("Flow rates did not converge")

        optimization.xn = xn

        return optimization


def get_default_single_pipeline_model():
    model = MultiPipelineModel(name="Single Pipe Model")
    pipe1 = Pipeline(500, 0.05, 60, name="Pipeline 1")
    model.pipelines.append(pipe1)
    return model


def get_default_dual_pipeline_model():
    model = MultiPipelineModel(name="Dual Pipe Model")
    model.pipelines.append(Pipeline(500, 0.05, 60, name="Pipeline 1"))
    model.pipelines.append(Pipeline(2500, 0.03, 45, name="Pipeline 2"))
    return model


def get_default_tri_pipeline_model():
    model = MultiPipelineModel(name="Tri Pipe Model")
    model.pipelines.append(Pipeline(500, 0.05, 40, name="Pipeline 1"))
    model.pipelines.append(Pipeline(1200, 0.03, 45, name="Pipeline 2"))
    model.pipelines.append(Pipeline(1000, 0.02, 12, name="Pipeline 3"))
    return model


def get_default_quad_pipeline_model():
    model = MultiPipelineModel(name="Quad Pipe Model")
    model.pipelines.append(Pipeline(750, 0.05, 40, name="Pipeline 1"))
    model.pipelines.append(Pipeline(1200, 0.03, 45, name="Pipeline 2"))
    model.pipelines.append(Pipeline(7000, 0.02, 42, name="Pipeline 3"))
    model.pipelines.append(Pipeline(2000, 0.06, 47, name="Pipeline 4"))
    return model
