// -----------------------------------------------------------------------------
// sa_accel.v - Systolic-array matrix-multiply accelerator (top level)
//
// Computes one tile:  C (N x N, int32) = A (N x K, int8) * B (K x N, int8),
// with 1 <= K <= KMAX. Larger matrices are tiled by the software driver.
//
// Contains:
//   * A and B operand buffers (written over the register bus)
//   * a controller FSM that skews/streams operands into the array
//   * a performance counter (cycles spent running)
//   * a simple memory-mapped register bus (word addressed)
//
// Register map (word addresses):
//   0x0000 CTRL    W   bit0 = start (ignored while busy)
//   0x0001 STATUS  R   bit0 = busy, bit1 = done, bit2 = error (bad K)
//   0x0002 K       RW  inner dimension, 1..KMAX
//   0x0003 CYCLES  R   cycles the last run spent in the array
//   0x0004 N       R   array size (build parameter)
//   0x0005 KMAX    R   buffer depth (build parameter)
//   0x1000 + i*KMAX + k   A[i][k]   W (int8 in bits [7:0])
//   0x2000 + k*N + j      B[k][j]   W (int8 in bits [7:0])
//   0x3000 + i*N + j      C[i][j]   R (int32)
//
// Requires N*KMAX <= 4096 so each buffer fits in its 4K-word window.
// -----------------------------------------------------------------------------
`timescale 1ns/1ps

module sa_accel #(
    parameter N    = 4,
    parameter KMAX = 64,
    parameter DW   = 8,
    parameter AW   = 32
) (
    input  wire        clk,
    input  wire        rst,

    // register bus: synchronous write, combinational read
    input  wire        bus_we,
    input  wire [15:0] bus_addr,
    input  wire [31:0] bus_wdata,
    output reg  [31:0] bus_rdata
);
    localparam S_IDLE  = 2'd0;
    localparam S_CLEAR = 2'd1;
    localparam S_RUN   = 2'd2;
    localparam S_DONE  = 2'd3;

    reg [1:0]  state;
    reg [15:0] k_reg;     // inner dimension
    reg [15:0] t;         // cycle index inside a run
    reg [15:0] cycles;    // performance counter
    reg        err;

    reg signed [DW-1:0] a_mem [0:N*KMAX-1];   // A[i][k] at i*KMAX + k
    reg signed [DW-1:0] b_mem [0:KMAX*N-1];   // B[k][j] at k*N + j

    // ------------------------------------------------------------------ array
    wire              clear = (state == S_CLEAR);
    wire              run   = (state == S_RUN);
    wire [N*DW-1:0]   a_edge;
    wire [N*DW-1:0]   b_edge;
    wire [N*N*AW-1:0] acc_flat;

    // Skewed operand streaming: row i of A and column i of B are each delayed
    // by i cycles, so PE(i,j) sees A[i][k] and B[k][j] together at t = k+i+j.
    genvar r;
    generate
        for (r = 0; r < N; r = r + 1) begin : g_stream
            wire valid = run && (t >= r) && ((t - r) < k_reg);
            assign a_edge[r*DW +: DW] = valid ? a_mem[r*KMAX + (t - r)] : {DW{1'b0}};
            assign b_edge[r*DW +: DW] = valid ? b_mem[(t - r)*N + r]    : {DW{1'b0}};
        end
    endgenerate

    systolic_array #(.N(N), .DW(DW), .AW(AW)) u_array (
        .clk      (clk),
        .rst      (rst),
        .clear    (clear),
        .a_edge   (a_edge),
        .b_edge   (b_edge),
        .acc_flat (acc_flat)
    );

    // ------------------------------------------------- bus writes + controller
    always @(posedge clk) begin
        if (rst) begin
            state  <= S_IDLE;
            k_reg  <= 16'd0;
            t      <= 16'd0;
            cycles <= 16'd0;
            err    <= 1'b0;
        end else begin
            // ---- register-bus writes
            if (bus_we) begin
                case (bus_addr[15:12])
                    4'h0: begin
                        if (bus_addr[11:0] == 12'h002 && (state == S_IDLE || state == S_DONE))
                            k_reg <= bus_wdata[15:0];
                        if (bus_addr[11:0] == 12'h000 && bus_wdata[0] &&
                            (state == S_IDLE || state == S_DONE)) begin
                            if (k_reg >= 1 && k_reg <= KMAX) begin
                                state <= S_CLEAR;
                                err   <= 1'b0;
                            end else begin
                                err   <= 1'b1;
                            end
                        end
                    end
                    4'h1: if (bus_addr[11:0] < N*KMAX)
                              a_mem[bus_addr[11:0]] <= bus_wdata[DW-1:0];
                    4'h2: if (bus_addr[11:0] < KMAX*N)
                              b_mem[bus_addr[11:0]] <= bus_wdata[DW-1:0];
                    default: ;
                endcase
            end

            // ---- controller FSM
            case (state)
                S_CLEAR: begin
                    state  <= S_RUN;
                    t      <= 16'd0;
                    cycles <= 16'd0;
                end
                S_RUN: begin
                    t      <= t + 16'd1;
                    cycles <= cycles + 16'd1;
                    if (t == k_reg + 2*N - 3)
                        state <= S_DONE;
                end
                default: ;
            endcase
        end
    end

    // -------------------------------------------------------------- bus reads
    always @(*) begin
        bus_rdata = 32'd0;
        case (bus_addr[15:12])
            4'h0: begin
                case (bus_addr[11:0])
                    12'h001: bus_rdata = {29'd0, err, (state == S_DONE),
                                          (state == S_CLEAR || state == S_RUN)};
                    12'h002: bus_rdata = {16'd0, k_reg};
                    12'h003: bus_rdata = {16'd0, cycles};
                    12'h004: bus_rdata = N;
                    12'h005: bus_rdata = KMAX;
                    default: ;
                endcase
            end
            4'h3: if (bus_addr[11:0] < N*N)
                      bus_rdata = acc_flat[bus_addr[11:0]*AW +: AW];
            default: ;
        endcase
    end
endmodule
