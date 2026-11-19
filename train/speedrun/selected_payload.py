import math
import triton
import triton.language as tl

@triton.jit
def _flat23(Index, K: tl.constexpr, B: tl.constexpr, I0, O0, S0_0: tl.constexpr, S0_1: tl.constexpr, S0_2: tl.constexpr,
    S0_3: tl.constexpr, D0_1: tl.constexpr, D0_2: tl.constexpr, D0_3: tl.constexpr, N0: tl.constexpr, R0: tl.constexpr,
    E0: tl.constexpr, I1, O1, S1_0: tl.constexpr, S1_1: tl.constexpr, S1_2: tl.constexpr, S1_3: tl.constexpr,
    D1_1: tl.constexpr, D1_2: tl.constexpr, D1_3: tl.constexpr, N1: tl.constexpr, R1: tl.constexpr, E1: tl.constexpr,
    I2, O2, S2_0: tl.constexpr, S2_1: tl.constexpr, S2_2: tl.constexpr, S2_3: tl.constexpr, D2_1: tl.constexpr,
    D2_2: tl.constexpr, D2_3: tl.constexpr, N2: tl.constexpr, R2: tl.constexpr, E2: tl.constexpr, I3, O3,
    S3_0: tl.constexpr, S3_1: tl.constexpr, S3_2: tl.constexpr, S3_3: tl.constexpr, D3_1: tl.constexpr,
    D3_2: tl.constexpr, D3_3: tl.constexpr, N3: tl.constexpr, R3: tl.constexpr, E3: tl.constexpr, I4, O4,
    S4_0: tl.constexpr, S4_1: tl.constexpr, S4_2: tl.constexpr, S4_3: tl.constexpr, D4_1: tl.constexpr,
    D4_2: tl.constexpr, D4_3: tl.constexpr, N4: tl.constexpr, R4: tl.constexpr, E4: tl.constexpr, I5, O5,
    S5_0: tl.constexpr, S5_1: tl.constexpr, S5_2: tl.constexpr, S5_3: tl.constexpr, D5_1: tl.constexpr,
    D5_2: tl.constexpr, D5_3: tl.constexpr, N5: tl.constexpr, R5: tl.constexpr, E5: tl.constexpr, I6, O6,
    S6_0: tl.constexpr, S6_1: tl.constexpr, S6_2: tl.constexpr, S6_3: tl.constexpr, D6_1: tl.constexpr,
    D6_2: tl.constexpr, D6_3: tl.constexpr, N6: tl.constexpr, R6: tl.constexpr, E6: tl.constexpr, I7, O7,
    S7_0: tl.constexpr, S7_1: tl.constexpr, S7_2: tl.constexpr, S7_3: tl.constexpr, D7_1: tl.constexpr,
    D7_2: tl.constexpr, D7_3: tl.constexpr, N7: tl.constexpr, R7: tl.constexpr, E7: tl.constexpr, I8, O8,
    S8_0: tl.constexpr, S8_1: tl.constexpr, S8_2: tl.constexpr, S8_3: tl.constexpr, D8_1: tl.constexpr,
    D8_2: tl.constexpr, D8_3: tl.constexpr, N8: tl.constexpr, R8: tl.constexpr, E8: tl.constexpr, I9, O9,
    S9_0: tl.constexpr, S9_1: tl.constexpr, S9_2: tl.constexpr, S9_3: tl.constexpr, D9_1: tl.constexpr,
    D9_2: tl.constexpr, D9_3: tl.constexpr, N9: tl.constexpr, R9: tl.constexpr, E9: tl.constexpr, I10, O10,
    S10_0: tl.constexpr, S10_1: tl.constexpr, S10_2: tl.constexpr, S10_3: tl.constexpr, D10_1: tl.constexpr,
    D10_2: tl.constexpr, D10_3: tl.constexpr, N10: tl.constexpr, R10: tl.constexpr, E10: tl.constexpr, I11, O11,
    S11_0: tl.constexpr, S11_1: tl.constexpr, S11_2: tl.constexpr, S11_3: tl.constexpr, D11_1: tl.constexpr,
    D11_2: tl.constexpr, D11_3: tl.constexpr, N11: tl.constexpr, R11: tl.constexpr, E11: tl.constexpr, I12, O12,
    S12_0: tl.constexpr, S12_1: tl.constexpr, S12_2: tl.constexpr, S12_3: tl.constexpr, D12_1: tl.constexpr,
    D12_2: tl.constexpr, D12_3: tl.constexpr, N12: tl.constexpr, R12: tl.constexpr, E12: tl.constexpr, I13, O13,
    S13_0: tl.constexpr, S13_1: tl.constexpr, S13_2: tl.constexpr, S13_3: tl.constexpr, D13_1: tl.constexpr,
    D13_2: tl.constexpr, D13_3: tl.constexpr, N13: tl.constexpr, R13: tl.constexpr, E13: tl.constexpr, I14, O14,
    S14_0: tl.constexpr, S14_1: tl.constexpr, S14_2: tl.constexpr, S14_3: tl.constexpr, D14_1: tl.constexpr,
    D14_2: tl.constexpr, D14_3: tl.constexpr, N14: tl.constexpr, R14: tl.constexpr, E14: tl.constexpr, I15, O15,
    S15_0: tl.constexpr, S15_1: tl.constexpr, S15_2: tl.constexpr, S15_3: tl.constexpr, D15_1: tl.constexpr,
    D15_2: tl.constexpr, D15_3: tl.constexpr, N15: tl.constexpr, R15: tl.constexpr, E15: tl.constexpr, I16, O16,
    S16_0: tl.constexpr, S16_1: tl.constexpr, S16_2: tl.constexpr, S16_3: tl.constexpr, D16_1: tl.constexpr,
    D16_2: tl.constexpr, D16_3: tl.constexpr, N16: tl.constexpr, R16: tl.constexpr, E16: tl.constexpr, I17, O17,
    S17_0: tl.constexpr, S17_1: tl.constexpr, S17_2: tl.constexpr, S17_3: tl.constexpr, D17_1: tl.constexpr,
    D17_2: tl.constexpr, D17_3: tl.constexpr, N17: tl.constexpr, R17: tl.constexpr, E17: tl.constexpr, I18, O18,
    S18_0: tl.constexpr, S18_1: tl.constexpr, S18_2: tl.constexpr, S18_3: tl.constexpr, D18_1: tl.constexpr,
    D18_2: tl.constexpr, D18_3: tl.constexpr, N18: tl.constexpr, R18: tl.constexpr, E18: tl.constexpr, I19, O19,
    S19_0: tl.constexpr, S19_1: tl.constexpr, S19_2: tl.constexpr, S19_3: tl.constexpr, D19_1: tl.constexpr,
    D19_2: tl.constexpr, D19_3: tl.constexpr, N19: tl.constexpr, R19: tl.constexpr, E19: tl.constexpr, I20, O20,
    S20_0: tl.constexpr, S20_1: tl.constexpr, S20_2: tl.constexpr, S20_3: tl.constexpr, D20_1: tl.constexpr,
    D20_2: tl.constexpr, D20_3: tl.constexpr, N20: tl.constexpr, R20: tl.constexpr, E20: tl.constexpr, I21, O21,
    S21_0: tl.constexpr, S21_1: tl.constexpr, S21_2: tl.constexpr, S21_3: tl.constexpr, D21_1: tl.constexpr,
    D21_2: tl.constexpr, D21_3: tl.constexpr, N21: tl.constexpr, R21: tl.constexpr, E21: tl.constexpr, I22, O22,
    S22_0: tl.constexpr, S22_1: tl.constexpr, S22_2: tl.constexpr, S22_3: tl.constexpr, D22_1: tl.constexpr,
    D22_2: tl.constexpr, D22_3: tl.constexpr, N22: tl.constexpr, R22: tl.constexpr, E22: tl.constexpr):
    pid = tl.program_id(0)
    if (pid >= 0) & (pid < E0):
        p0 = (pid - 0) * B + tl.arange(0, B)
        row0 = p0 // N0
        rem0 = p0 % N0
        selected0 = tl.load(Index + row0, row0 < K, 0)
        source0 = selected0 * S0_0 + rem0 // (D0_2 * D0_3) % D0_1 * S0_1 + rem0 // D0_3 % D0_2 * S0_2 + rem0 % D0_3 * S0_3
        value0 = tl.load(I0 + source0, p0 < K * N0, 0)
        tl.store(O0 + p0, value0, p0 < K * N0)
    if (pid >= E0) & (pid < E1):
        p1 = (pid - E0) * B + tl.arange(0, B)
        row1 = p1 // N1
        rem1 = p1 % N1
        selected1 = tl.load(Index + row1, row1 < K, 0)
        source1 = selected1 * S1_0 + rem1 // (D1_2 * D1_3) % D1_1 * S1_1 + rem1 // D1_3 % D1_2 * S1_2 + rem1 % D1_3 * S1_3
        value1 = tl.load(I1 + source1, p1 < K * N1, 0)
        tl.store(O1 + p1, value1, p1 < K * N1)
    if (pid >= E1) & (pid < E2):
        p2 = (pid - E1) * B + tl.arange(0, B)
        row2 = p2 // N2
        rem2 = p2 % N2
        selected2 = tl.load(Index + row2, row2 < K, 0)
        source2 = selected2 * S2_0 + rem2 // (D2_2 * D2_3) % D2_1 * S2_1 + rem2 // D2_3 % D2_2 * S2_2 + rem2 % D2_3 * S2_3
        value2 = tl.load(I2 + source2, p2 < K * N2, 0)
        tl.store(O2 + p2, value2, p2 < K * N2)
    if (pid >= E2) & (pid < E3):
        p3 = (pid - E2) * B + tl.arange(0, B)
        row3 = p3 // N3
        rem3 = p3 % N3
        selected3 = tl.load(Index + row3, row3 < K, 0)
        source3 = selected3 * S3_0 + rem3 // (D3_2 * D3_3) % D3_1 * S3_1 + rem3 // D3_3 % D3_2 * S3_2 + rem3 % D3_3 * S3_3
        value3 = tl.load(I3 + source3, p3 < K * N3, 0)
        tl.store(O3 + p3, value3, p3 < K * N3)
    if (pid >= E3) & (pid < E4):
        p4 = (pid - E3) * B + tl.arange(0, B)
        row4 = p4 // N4
        rem4 = p4 % N4
        selected4 = tl.load(Index + row4, row4 < K, 0)
        source4 = selected4 * S4_0 + rem4 // (D4_2 * D4_3) % D4_1 * S4_1 + rem4 // D4_3 % D4_2 * S4_2 + rem4 % D4_3 * S4_3
        value4 = tl.load(I4 + source4, p4 < K * N4, 0)
        tl.store(O4 + p4, value4, p4 < K * N4)
    if (pid >= E4) & (pid < E5):
        p5 = (pid - E4) * B + tl.arange(0, B)
        row5 = p5 // N5
        rem5 = p5 % N5
        selected5 = tl.load(Index + row5, row5 < K, 0)
        source5 = selected5 * S5_0 + rem5 // (D5_2 * D5_3) % D5_1 * S5_1 + rem5 // D5_3 % D5_2 * S5_2 + rem5 % D5_3 * S5_3
        value5 = tl.load(I5 + source5, p5 < K * N5, 0)
        tl.store(O5 + p5, value5, p5 < K * N5)
    if (pid >= E5) & (pid < E6):
        p6 = (pid - E5) * B + tl.arange(0, B)
        row6 = p6 // N6
        rem6 = p6 % N6
        selected6 = tl.load(Index + row6, row6 < K, 0)
        source6 = selected6 * S6_0 + rem6 // (D6_2 * D6_3) % D6_1 * S6_1 + rem6 // D6_3 % D6_2 * S6_2 + rem6 % D6_3 * S6_3
        value6 = tl.load(I6 + source6, p6 < K * N6, 0)
        tl.store(O6 + p6, value6, p6 < K * N6)
    if (pid >= E6) & (pid < E7):
        p7 = (pid - E6) * B + tl.arange(0, B)
        row7 = p7 // N7
        rem7 = p7 % N7
        selected7 = tl.load(Index + row7, row7 < K, 0)
        source7 = selected7 * S7_0 + rem7 // (D7_2 * D7_3) % D7_1 * S7_1 + rem7 // D7_3 % D7_2 * S7_2 + rem7 % D7_3 * S7_3
        value7 = tl.load(I7 + source7, p7 < K * N7, 0)
        tl.store(O7 + p7, value7, p7 < K * N7)
    if (pid >= E7) & (pid < E8):
        p8 = (pid - E7) * B + tl.arange(0, B)
        row8 = p8 // N8
        rem8 = p8 % N8
        selected8 = tl.load(Index + row8, row8 < K, 0)
        source8 = selected8 * S8_0 + rem8 // (D8_2 * D8_3) % D8_1 * S8_1 + rem8 // D8_3 % D8_2 * S8_2 + rem8 % D8_3 * S8_3
        value8 = tl.load(I8 + source8, p8 < K * N8, 0)
        tl.store(O8 + p8, value8, p8 < K * N8)
    if (pid >= E8) & (pid < E9):
        p9 = (pid - E8) * B + tl.arange(0, B)
        row9 = p9 // N9
        rem9 = p9 % N9
        selected9 = tl.load(Index + row9, row9 < K, 0)
        source9 = selected9 * S9_0 + rem9 // (D9_2 * D9_3) % D9_1 * S9_1 + rem9 // D9_3 % D9_2 * S9_2 + rem9 % D9_3 * S9_3
        value9 = tl.load(I9 + source9, p9 < K * N9, 0)
        tl.store(O9 + p9, value9, p9 < K * N9)
    if (pid >= E9) & (pid < E10):
        p10 = (pid - E9) * B + tl.arange(0, B)
        row10 = p10 // N10
        rem10 = p10 % N10
        selected10 = tl.load(Index + row10, row10 < K, 0)
        source10 = selected10 * S10_0 + rem10 // (D10_2 * D10_3) % D10_1 * S10_1 + rem10 // D10_3 % D10_2 * S10_2 + rem10 % D10_3 * S10_3
        value10 = tl.load(I10 + source10, p10 < K * N10, 0)
        tl.store(O10 + p10, value10, p10 < K * N10)
    if (pid >= E10) & (pid < E11):
        p11 = (pid - E10) * B + tl.arange(0, B)
        row11 = p11 // N11
        rem11 = p11 % N11
        selected11 = tl.load(Index + row11, row11 < K, 0)
        source11 = selected11 * S11_0 + rem11 // (D11_2 * D11_3) % D11_1 * S11_1 + rem11 // D11_3 % D11_2 * S11_2 + rem11 % D11_3 * S11_3
        value11 = tl.load(I11 + source11, p11 < K * N11, 0)
        tl.store(O11 + p11, value11, p11 < K * N11)
    if (pid >= E11) & (pid < E12):
        p12 = (pid - E11) * B + tl.arange(0, B)
        row12 = p12 // N12
        rem12 = p12 % N12
        selected12 = tl.load(Index + row12, row12 < K, 0)
        source12 = selected12 * S12_0 + rem12 // (D12_2 * D12_3) % D12_1 * S12_1 + rem12 // D12_3 % D12_2 * S12_2 + rem12 % D12_3 * S12_3
        value12 = tl.load(I12 + source12, p12 < K * N12, 0)
        tl.store(O12 + p12, value12, p12 < K * N12)
    if (pid >= E12) & (pid < E13):
        p13 = (pid - E12) * B + tl.arange(0, B)
        row13 = p13 // N13
        rem13 = p13 % N13
        selected13 = tl.load(Index + row13, row13 < K, 0)
        source13 = selected13 * S13_0 + rem13 // (D13_2 * D13_3) % D13_1 * S13_1 + rem13 // D13_3 % D13_2 * S13_2 + rem13 % D13_3 * S13_3
        value13 = tl.load(I13 + source13, p13 < K * N13, 0)
        tl.store(O13 + p13, value13, p13 < K * N13)
    if (pid >= E13) & (pid < E14):
        p14 = (pid - E13) * B + tl.arange(0, B)
        row14 = p14 // N14
        rem14 = p14 % N14
        selected14 = tl.load(Index + row14, row14 < K, 0)
        source14 = selected14 * S14_0 + rem14 // (D14_2 * D14_3) % D14_1 * S14_1 + rem14 // D14_3 % D14_2 * S14_2 + rem14 % D14_3 * S14_3
        value14 = tl.load(I14 + source14, p14 < K * N14, 0)
        tl.store(O14 + p14, value14, p14 < K * N14)
    if (pid >= E14) & (pid < E15):
        p15 = (pid - E14) * B + tl.arange(0, B)
        row15 = p15 // N15
        rem15 = p15 % N15
        selected15 = tl.load(Index + row15, row15 < K, 0)
        source15 = selected15 * S15_0 + rem15 // (D15_2 * D15_3) % D15_1 * S15_1 + rem15 // D15_3 % D15_2 * S15_2 + rem15 % D15_3 * S15_3
        value15 = tl.load(I15 + source15, p15 < K * N15, 0)
        tl.store(O15 + p15, value15, p15 < K * N15)
    if (pid >= E15) & (pid < E16):
        p16 = (pid - E15) * B + tl.arange(0, B)
        row16 = p16 // N16
        rem16 = p16 % N16
        selected16 = tl.load(Index + row16, row16 < K, 0)
        source16 = selected16 * S16_0 + rem16 // (D16_2 * D16_3) % D16_1 * S16_1 + rem16 // D16_3 % D16_2 * S16_2 + rem16 % D16_3 * S16_3
        value16 = tl.load(I16 + source16, p16 < K * N16, 0)
        tl.store(O16 + p16, value16, p16 < K * N16)
    if (pid >= E16) & (pid < E17):
        p17 = (pid - E16) * B + tl.arange(0, B)
        row17 = p17 // N17
        rem17 = p17 % N17
        selected17 = tl.load(Index + row17, row17 < K, 0)
        source17 = selected17 * S17_0 + rem17 // (D17_2 * D17_3) % D17_1 * S17_1 + rem17 // D17_3 % D17_2 * S17_2 + rem17 % D17_3 * S17_3
        value17 = tl.load(I17 + source17, p17 < K * N17, 0)
        tl.store(O17 + p17, value17, p17 < K * N17)
    if (pid >= E17) & (pid < E18):
        p18 = (pid - E17) * B + tl.arange(0, B)
        row18 = p18 // N18
        rem18 = p18 % N18
        selected18 = tl.load(Index + row18, row18 < K, 0)
        source18 = selected18 * S18_0 + rem18 // (D18_2 * D18_3) % D18_1 * S18_1 + rem18 // D18_3 % D18_2 * S18_2 + rem18 % D18_3 * S18_3
        value18 = tl.load(I18 + source18, p18 < K * N18, 0)
        tl.store(O18 + p18, value18, p18 < K * N18)
    if (pid >= E18) & (pid < E19):
        p19 = (pid - E18) * B + tl.arange(0, B)
        row19 = p19 // N19
        rem19 = p19 % N19
        selected19 = tl.load(Index + row19, row19 < K, 0)
        source19 = selected19 * S19_0 + rem19 // (D19_2 * D19_3) % D19_1 * S19_1 + rem19 // D19_3 % D19_2 * S19_2 + rem19 % D19_3 * S19_3
        value19 = tl.load(I19 + source19, p19 < K * N19, 0)
        tl.store(O19 + p19, value19, p19 < K * N19)
    if (pid >= E19) & (pid < E20):
        p20 = (pid - E19) * B + tl.arange(0, B)
        row20 = p20 // N20
        rem20 = p20 % N20
        selected20 = tl.load(Index + row20, row20 < K, 0)
        source20 = selected20 * S20_0 + rem20 // (D20_2 * D20_3) % D20_1 * S20_1 + rem20 // D20_3 % D20_2 * S20_2 + rem20 % D20_3 * S20_3
        value20 = tl.load(I20 + source20, p20 < K * N20, 0)
        tl.store(O20 + p20, value20, p20 < K * N20)
    if (pid >= E20) & (pid < E21):
        p21 = (pid - E20) * B + tl.arange(0, B)
        row21 = p21 // N21
        rem21 = p21 % N21
        selected21 = tl.load(Index + row21, row21 < K, 0)
        source21 = selected21 * S21_0 + rem21 // (D21_2 * D21_3) % D21_1 * S21_1 + rem21 // D21_3 % D21_2 * S21_2 + rem21 % D21_3 * S21_3
        value21 = tl.load(I21 + source21, p21 < K * N21, 0)
        tl.store(O21 + p21, value21, p21 < K * N21)
    if (pid >= E21) & (pid < E22):
        p22 = (pid - E21) * B + tl.arange(0, B)
        row22 = p22 // N22
        rem22 = p22 % N22
        selected22 = tl.load(Index + row22, row22 < K, 0)
        source22 = selected22 * S22_0 + rem22 // (D22_2 * D22_3) % D22_1 * S22_1 + rem22 // D22_3 % D22_2 * S22_2 + rem22 % D22_3 * S22_3
        value22 = tl.load(I22 + source22, p22 < K * N22, 0)
        tl.store(O22 + p22, value22, p22 < K * N22)


def launch(inputs, index, block=1024):
    assert len(inputs) == 23
    k = index.numel()
    outputs = [z.new_empty((k, *z.shape[1:])) for z in inputs]
    args = [index, k, block]
    end = 0
    for z, out in zip(inputs, outputs):
        assert 1 <= z.ndim <= 4
        dims = (*z.shape[1:], *((1,) * (4 - z.ndim)))
        strides = (*z.stride(), *((0,) * (4 - z.ndim)))
        size = math.prod(dims)
        end += triton.cdiv(k * size, block)
        args.extend((z, out, *strides, *dims, size, triton.next_power_of_2(size), end))
    _flat23[(end,)](*args, num_warps=4)
    return outputs


def selected_payload(mids, xL, routes, signed_steps, base, target_w, y, moments, control, inside, start, length, site):
    fields = []
    def put(z, local=False):
        if z is None:
            return None
        fields.append(z if local else z[start:start + length])
        return len(fields) - 1
    ms = [put(z) if b >= site else None for b, z in enumerate(mids)]
    xl = put(xL)
    rs = [put(z[0]) if b >= site else None for b, z in enumerate(routes)]
    cs = [put(z[1]) if b >= site else None for b, z in enumerate(routes)]
    st = put(signed_steps)
    bs = put(base)
    tw = put(target_w)
    yy = put(y)
    mm = [put(z) for z in moments]
    ct = put(control, True)
    assert len(fields) == 23
    chosen = launch(fields, inside)
    def get(i):
        return None if i is None else chosen[i]
    return ([get(i) for i in ms], get(xl), [get(i) for i in rs], [get(i) for i in cs],
            get(st), get(bs), get(tw), get(yy), tuple(get(i) for i in mm), get(ct))
