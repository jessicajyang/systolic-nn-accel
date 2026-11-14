// -----------------------------------------------------------------------------
// systolic_array.v - N x N output-stationary systolic array
//
//   a_edge[i] enters row i from the left, b_edge[j] enters column j from the top.
//   The caller must skew the inputs: row i is delayed i cycles, column j is
//   delayed j cycles. PE(i,j) then sees A[i][k] and B[k][j] together at
//   cycle t = k + i + j.
//
//   For a K-long dot product the last PE finishes at t = K + 2N - 3, so a run
//   takes K + 2N - 2 cycles (3N - 2 when K = N).
// -----------------------------------------------------------------------------
`timescale 1ns/1ps

module systolic_array #(
    parameter N  = 4,
    parameter DW = 8,
    parameter AW = 32
) (
    input  wire                clk,
    input  wire                rst,
    input  wire                clear,
    input  wire [N*DW-1:0]     a_edge,    // one DW-bit lane per row
    input  wire [N*DW-1:0]     b_edge,    // one DW-bit lane per column
    output wire [N*N*AW-1:0]   acc_flat   // C[i][j] at index i*N + j
);
    // Operand wires between PEs (flattened to stay plain Verilog-2001).
    //   a_w index (i*(N+1) + j): value entering PE(i,j) from the left
    //   b_w index (i*N + j)    : value entering PE(i,j) from the top
    wire [N*(N+1)*DW-1:0] a_w;
    wire [(N+1)*N*DW-1:0] b_w;

    genvar i, j;
    generate
        for (i = 0; i < N; i = i + 1) begin : g_a_edge
            assign a_w[(i*(N+1))*DW +: DW] = a_edge[i*DW +: DW];
        end
        for (j = 0; j < N; j = j + 1) begin : g_b_edge
            assign b_w[j*DW +: DW] = b_edge[j*DW +: DW];
        end

        for (i = 0; i < N; i = i + 1) begin : g_row
            for (j = 0; j < N; j = j + 1) begin : g_col
                pe #(.DW(DW), .AW(AW)) u_pe (
                    .clk   (clk),
                    .rst   (rst),
                    .clear (clear),
                    .a_in  (a_w[(i*(N+1) + j)*DW +: DW]),
                    .b_in  (b_w[(i*N + j)*DW +: DW]),
                    .a_out (a_w[(i*(N+1) + j + 1)*DW +: DW]),
                    .b_out (b_w[((i+1)*N + j)*DW +: DW]),
                    .acc   (acc_flat[(i*N + j)*AW +: AW])
                );
            end
        end
    endgenerate
endmodule
