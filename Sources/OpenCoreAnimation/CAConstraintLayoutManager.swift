//
//  CAConstraintLayoutManager.swift
//  OpenCoreAnimation
//
//  Internal delegate protocol for rendering layer trees.
//

import Foundation


/// An object that provides a constraint-based layout manager.
open class CAConstraintLayoutManager: CALayoutManager {
    public init() {}

    // MARK: - CALayoutManager

    public func invalidateLayout(of layer: CALayer) {
        layer.setNeedsLayout()
    }

    public func layoutSublayers(of layer: CALayer) {
        guard let sublayers = layer.sublayers, !sublayers.isEmpty else { return }

        let layerIndices = Dictionary(
            uniqueKeysWithValues: sublayers.enumerated().map { (ObjectIdentifier($0.element), $0.offset) }
        )
        let namedLayers = sublayers.reduce(into: [String: Int]()) { result, sublayer in
            guard let name = sublayer.name,
                  result[name] == nil,
                  let index = layerIndices[ObjectIdentifier(sublayer)] else { return }
            result[name] = index
        }
        let initialValues = sublayers.flatMap(Self.geometryValues)
        let superlayerValues = Self.superlayerGeometryValues(layer)

        var equations: [Equation] = []
        for (targetIndex, sublayer) in sublayers.enumerated() {
            for constraint in sublayer.constraints ?? [] {
                guard let equation = equation(
                    for: constraint,
                    targetLayerIndex: targetIndex,
                    namedLayers: namedLayers,
                    superlayerValues: superlayerValues
                ) else {
                    continue
                }
                equations.append(equation)
            }
        }
        guard !equations.isEmpty else { return }

        let solvedValues = solveIndependentComponents(
            equations: equations,
            initialValues: initialValues
        )
        apply(solvedValues, to: sublayers)
    }

    public func preferredSize(of layer: CALayer) -> CGSize {
        return layer.bounds.size
    }

    // MARK: - Linear System

    private struct Equation {
        var coefficients: [Int: CGFloat]
        var rightHandSide: CGFloat
        var involvedVariables: Set<Int>
    }

    private struct ReducedSystem {
        var rank: Int
        var isConsistent: Bool
        var solution: [CGFloat]?
    }

    private static let variablesPerLayer = 4
    private static let minimumXOffset = 0
    private static let widthOffset = 1
    private static let minimumYOffset = 2
    private static let heightOffset = 3
    private static let eliminationTolerance: CGFloat = 1e-10

    private func equation(
        for constraint: CAConstraint,
        targetLayerIndex: Int,
        namedLayers: [String: Int],
        superlayerValues: [CGFloat]
    ) -> Equation? {
        let targetTerms = Self.attributeTerms(
            constraint.attribute,
            layerIndex: targetLayerIndex
        )
        var coefficients = targetTerms
        var involvedVariables = Set(targetTerms.keys)
        var rightHandSide = constraint.offset

        if constraint.sourceName == "superlayer" {
            rightHandSide += constraint.scale
                * Self.attributeValue(constraint.sourceAttribute, values: superlayerValues)
        } else {
            guard let sourceIndex = namedLayers[constraint.sourceName] else { return nil }
            let sourceTerms = Self.attributeTerms(
                constraint.sourceAttribute,
                layerIndex: sourceIndex
            )
            involvedVariables.formUnion(sourceTerms.keys)
            for (index, coefficient) in sourceTerms {
                coefficients[index, default: 0] -= constraint.scale * coefficient
                if abs(coefficients[index, default: 0]) <= Self.eliminationTolerance {
                    coefficients.removeValue(forKey: index)
                }
            }
        }

        guard rightHandSide.isFinite,
              coefficients.values.allSatisfy(\.isFinite) else {
            return nil
        }
        return Equation(
            coefficients: coefficients,
            rightHandSide: rightHandSide,
            involvedVariables: involvedVariables
        )
    }

    private func solveIndependentComponents(
        equations: [Equation],
        initialValues: [CGFloat]
    ) -> [CGFloat] {
        var parent = Array(initialValues.indices)

        func root(of index: Int) -> Int {
            var current = index
            while parent[current] != current {
                current = parent[current]
            }
            return current
        }

        func union(_ lhs: Int, _ rhs: Int) {
            let lhsRoot = root(of: lhs)
            let rhsRoot = root(of: rhs)
            if lhsRoot != rhsRoot {
                parent[rhsRoot] = lhsRoot
            }
        }

        for equation in equations {
            guard let first = equation.involvedVariables.first else { continue }
            for variable in equation.involvedVariables.dropFirst() {
                union(first, variable)
            }
        }

        var equationsByRoot: [Int: [Equation]] = [:]
        var variablesByRoot: [Int: Set<Int>] = [:]
        for equation in equations {
            guard let first = equation.involvedVariables.first else { continue }
            let componentRoot = root(of: first)
            equationsByRoot[componentRoot, default: []].append(equation)
            variablesByRoot[componentRoot, default: []].formUnion(equation.involvedVariables)
        }

        var result = initialValues
        for (componentRoot, componentEquations) in equationsByRoot {
            guard let variables = variablesByRoot[componentRoot] else { continue }
            solve(
                equations: componentEquations,
                variables: Array(variables),
                initialValues: initialValues,
                result: &result
            )
        }
        return result
    }

    private func solve(
        equations: [Equation],
        variables: [Int],
        initialValues: [CGFloat],
        result: inout [CGFloat]
    ) {
        let orderedVariables = variables.sorted()
        let localIndex = Dictionary(
            uniqueKeysWithValues: orderedVariables.enumerated().map { ($0.element, $0.offset) }
        )
        var rows = equations.map { equation -> [CGFloat] in
            var row = Array(repeating: CGFloat(0), count: orderedVariables.count + 1)
            for (globalIndex, coefficient) in equation.coefficients {
                guard let index = localIndex[globalIndex] else { continue }
                row[index] = coefficient
            }
            let initialContribution = equation.coefficients.reduce(CGFloat(0)) {
                $0 + $1.value * initialValues[$1.key]
            }
            row[orderedVariables.count] = equation.rightHandSide - initialContribution
            return row
        }

        var reduced = reduce(rows, variableCount: orderedVariables.count)
        guard reduced.isConsistent else { return }

        let defaultOrder = orderedVariables.sorted { lhs, rhs in
            let lhsPriority = Self.defaultEquationPriority(globalIndex: lhs)
            let rhsPriority = Self.defaultEquationPriority(globalIndex: rhs)
            if lhsPriority != rhsPriority { return lhsPriority < rhsPriority }
            return lhs < rhs
        }
        for globalIndex in defaultOrder where reduced.rank < orderedVariables.count {
            guard let index = localIndex[globalIndex] else { continue }
            var defaultRow = Array(repeating: CGFloat(0), count: orderedVariables.count + 1)
            defaultRow[index] = 1
            let candidateRows = rows + [defaultRow]
            let candidate = reduce(candidateRows, variableCount: orderedVariables.count)
            if candidate.isConsistent && candidate.rank > reduced.rank {
                rows = candidateRows
                reduced = candidate
            }
        }

        guard reduced.isConsistent,
              let solution = reduced.solution,
              solution.allSatisfy(\.isFinite) else {
            return
        }
        for (local, global) in orderedVariables.enumerated() {
            result[global] = initialValues[global] + solution[local]
        }
    }

    private func reduce(_ rows: [[CGFloat]], variableCount: Int) -> ReducedSystem {
        var matrix = rows
        var pivotRow = 0
        var pivotColumns: [Int] = []

        for column in 0..<variableCount where pivotRow < matrix.count {
            var selectedRow: Int?
            var selectedMagnitude = Self.eliminationTolerance
            for row in pivotRow..<matrix.count {
                let magnitude = abs(matrix[row][column])
                if magnitude > selectedMagnitude {
                    selectedMagnitude = magnitude
                    selectedRow = row
                }
            }
            guard let selectedRow else { continue }
            if selectedRow != pivotRow {
                matrix.swapAt(selectedRow, pivotRow)
            }

            let pivot = matrix[pivotRow][column]
            for index in column...variableCount {
                matrix[pivotRow][index] /= pivot
            }
            for row in matrix.indices where row != pivotRow {
                let factor = matrix[row][column]
                guard abs(factor) > Self.eliminationTolerance else { continue }
                for index in column...variableCount {
                    matrix[row][index] -= factor * matrix[pivotRow][index]
                }
            }
            pivotColumns.append(column)
            pivotRow += 1
        }

        let isConsistent = matrix.allSatisfy { row in
            let hasCoefficient = row[..<variableCount].contains {
                abs($0) > Self.eliminationTolerance
            }
            return hasCoefficient || abs(row[variableCount]) <= Self.eliminationTolerance
        }
        guard isConsistent, pivotColumns.count == variableCount else {
            return ReducedSystem(rank: pivotColumns.count, isConsistent: isConsistent, solution: nil)
        }

        var solution = Array(repeating: CGFloat(0), count: variableCount)
        for (row, column) in pivotColumns.enumerated() {
            solution[column] = matrix[row][variableCount]
        }
        return ReducedSystem(rank: pivotColumns.count, isConsistent: true, solution: solution)
    }

    private func apply(_ values: [CGFloat], to layers: [CALayer]) {
        for (index, layer) in layers.enumerated() {
            let base = index * Self.variablesPerLayer
            let minimumX = values[base + Self.minimumXOffset]
            let width = values[base + Self.widthOffset]
            let minimumY = values[base + Self.minimumYOffset]
            let height = values[base + Self.heightOffset]
            guard minimumX.isFinite,
                  minimumY.isFinite,
                  width.isFinite,
                  height.isFinite else {
                continue
            }

            var bounds = layer.bounds
            bounds.size = CGSize(width: width, height: height)
            let position = CGPoint(
                x: minimumX + width * layer.anchorPoint.x,
                y: minimumY + height * layer.anchorPoint.y
            )
            if layer.bounds != bounds { layer.bounds = bounds }
            if layer.position != position { layer.position = position }
        }
    }

    private static func geometryValues(for layer: CALayer) -> [CGFloat] {
        let width = layer.bounds.width
        let height = layer.bounds.height
        return [
            layer.position.x - width * layer.anchorPoint.x,
            width,
            layer.position.y - height * layer.anchorPoint.y,
            height,
        ]
    }

    private static func superlayerGeometryValues(_ layer: CALayer) -> [CGFloat] {
        [layer.bounds.minX, layer.bounds.width, layer.bounds.minY, layer.bounds.height]
    }

    private static func attributeTerms(
        _ attribute: CAConstraintAttribute,
        layerIndex: Int
    ) -> [Int: CGFloat] {
        let base = layerIndex * variablesPerLayer
        switch attribute {
        case .minX: return [base + minimumXOffset: 1]
        case .midX: return [base + minimumXOffset: 1, base + widthOffset: 0.5]
        case .maxX: return [base + minimumXOffset: 1, base + widthOffset: 1]
        case .width: return [base + widthOffset: 1]
        case .minY: return [base + minimumYOffset: 1]
        case .midY: return [base + minimumYOffset: 1, base + heightOffset: 0.5]
        case .maxY: return [base + minimumYOffset: 1, base + heightOffset: 1]
        case .height: return [base + heightOffset: 1]
        }
    }

    private static func attributeValue(
        _ attribute: CAConstraintAttribute,
        values: [CGFloat]
    ) -> CGFloat {
        switch attribute {
        case .minX: return values[minimumXOffset]
        case .midX: return values[minimumXOffset] + values[widthOffset] * 0.5
        case .maxX: return values[minimumXOffset] + values[widthOffset]
        case .width: return values[widthOffset]
        case .minY: return values[minimumYOffset]
        case .midY: return values[minimumYOffset] + values[heightOffset] * 0.5
        case .maxY: return values[minimumYOffset] + values[heightOffset]
        case .height: return values[heightOffset]
        }
    }

    private static func defaultEquationPriority(globalIndex: Int) -> Int {
        switch globalIndex % variablesPerLayer {
        case widthOffset, heightOffset: return 0
        default: return 1
        }
    }
}
