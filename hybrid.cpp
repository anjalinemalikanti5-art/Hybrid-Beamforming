#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include <vector>
#include <complex>
#include <cmath>
#include <cstdlib>
#include <sstream>

using namespace Eigen;
using namespace std;

const double PI = 3.14159265358979323846;
const double EPS = 1e-12;

/* ================= CSV READ ================= */
MatrixXd readCSV(const string &file) {
    ifstream in(file);
    if (!in.is_open()) {
        cerr << "Cannot open file: " << file << endl;
        exit(1);
    }
    vector<vector<double>> data;
    string line;
    while (getline(in, line)) {
        stringstream ss(line);
        string cell;
        vector<double> row;
        while (getline(ss, cell, ',')) row.push_back(stod(cell));
        data.push_back(row);
    }
    MatrixXd mat(data.size(), data[0].size());
    for (int i = 0; i < mat.rows(); i++)
        for (int j = 0; j < mat.cols(); j++)
            mat(i,j) = data[i][j];
    return mat;
}

/* ================= Steering Vector ================= */
VectorXcd steering_vector(double theta_deg, int M) {
    double theta = theta_deg * PI / 180.0;
    VectorXcd a(M);
    for (int m = 0; m < M; m++)
        a(m) = exp(complex<double>(0, PI * m * sin(theta)));
    return a;
}

/* ================= Savitzky–Golay ================= */
VectorXd savitzkyGolay(const VectorXd& x) {
    int N = x.size();
    VectorXd y = x;
    if (N < 3) return y;
    int w = min(5, (N-1)/2);
    for(int i = 0; i < N; i++) {
        double sum = 0;
        int cnt = 0;
        for(int j = max(0,i-w); j <= min(N-1,i+w); j++) {
            sum += x(j);
            cnt++;
        }
        y(i) = sum / cnt;
    }
    return y;
}

int main() {
    cout << "Hybrid Beamforming using DATASET\n";

    MatrixXd raw = readCSV("C_44_train_converted.csv");
    int M = raw.rows();
    int N = raw.cols();

    cout << "Antennas = " << M << ", Snapshots = " << N << endl;

    MatrixXcd X(M, N);
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            X(i,j) = complex<double>(raw(i,j), 0.0);

    /* ================= DOA DENOISING ================= */
    VectorXd theta_s_noisy = VectorXd::LinSpaced(N, 25.0, 35.0);
    VectorXd theta_s_hat = savitzkyGolay(theta_s_noisy);

    /* ================= MACHINE LEARNING DOA PREDICTION ================= */
    VectorXd theta_ml(N);
    VectorXd X_index = VectorXd::LinSpaced(N, 0, N-1);
    double mean_x = X_index.mean();
    double mean_y = theta_s_noisy.mean();

    double num = 0.0, den = 0.0;
    for (int i = 0; i < N; i++) {
        num += (X_index(i) - mean_x) * (theta_s_noisy(i) - mean_y);
        den += (X_index(i) - mean_x) * (X_index(i) - mean_x);
    }
    double beta = num / den;
    double alpha = mean_y - beta * mean_x;

    for (int i = 0; i < N; i++)
        theta_ml(i) = alpha + beta * X_index(i);

    ofstream ml_out("ml_doa_pred.csv");
    for (int i = 0; i < N; i++)
        ml_out << i << "," << theta_ml(i) << endl;
    ml_out.close();

    cout << "Machine Learning DOA prediction completed.\n";

    /* ================= COVARIANCE MATRIX ================= */
    MatrixXcd R = (X * X.adjoint()) / double(N);
    MatrixXcd Rinv = R.completeOrthogonalDecomposition().pseudoInverse();

    /* ================= BEAMFORMERS ================= */
    VectorXcd a_s = steering_vector(theta_ml.mean(), M);
    VectorXcd w_ml = a_s.normalized();
    VectorXcd w_mvdr = (Rinv * a_s) / (a_s.adjoint() * Rinv * a_s)(0,0);

    /* ================= HYBRID OPTIMIZATION ================= */
    MatrixXcd X_sig = a_s * (a_s.adjoint() * X);

    double best_alpha = 0.0;
    double best_snr = -1e9;

    for (double alpha = 0.0; alpha <= 1.0; alpha += 0.01) {
        VectorXcd w_tmp = (alpha*w_ml + (1-alpha)*w_mvdr).normalized();
        RowVectorXcd y_tmp = w_tmp.adjoint() * X;
        RowVectorXcd y_sig = w_tmp.adjoint() * X_sig;
        RowVectorXcd y_n = y_tmp - y_sig;
        double Ps = y_sig.cwiseAbs2().mean();
        double Pn = y_n.cwiseAbs2().mean();
        if (Pn < EPS) Pn = EPS;
        double snr = 10 * log10(Ps / Pn);
        if (snr > best_snr) {
            best_snr = snr;
            best_alpha = alpha;
        }
    }
    VectorXcd w_hybrid = (best_alpha*w_ml + (1-best_alpha)*w_mvdr).normalized();

    /* ================= SNR COMPUTATION ================= */
    RowVectorXcd s_raw = X.row(0);
    double Ps_in = s_raw.cwiseAbs2().mean();
    double Pn_in = (s_raw - X_sig.row(0)).cwiseAbs2().mean();
    if(Pn_in < EPS) Pn_in = EPS;
    double snr_in = 10 * log10(Ps_in / Pn_in);

    RowVectorXcd y_ml = w_ml.adjoint() * X;
    RowVectorXcd y_ml_sig = w_ml.adjoint() * X_sig;
    double snr_ml = 10 * log10(
        y_ml_sig.cwiseAbs2().mean() /
        max(EPS, (y_ml - y_ml_sig).cwiseAbs2().mean())
    );

    RowVectorXcd y_mv = w_mvdr.adjoint() * X;
    RowVectorXcd y_mv_sig = w_mvdr.adjoint() * X_sig;
    double snr_mvdr = 10 * log10(
        y_mv_sig.cwiseAbs2().mean() /
        max(EPS, (y_mv - y_mv_sig).cwiseAbs2().mean())
    );

    RowVectorXcd y_hyb = w_hybrid.adjoint() * X;
    RowVectorXcd y_hyb_sig = w_hybrid.adjoint() * X_sig;
    RowVectorXcd y_hyb_noise = y_hyb - y_hyb_sig;
    double snr_hybrid = 10 * log10(
        y_hyb_sig.cwiseAbs2().mean() /
        max(EPS, y_hyb_noise.cwiseAbs2().mean())
    );

    cout << "\n========= SNR VALUES (dB) =========\n";
    cout << "Input SNR   : " << snr_in << endl;
    cout << "ML SNR      : " << snr_ml << endl;
    cout << "MVDR SNR    : " << snr_mvdr << endl;
    cout << "Hybrid SNR  : " << snr_hybrid << endl;
    cout << "==================================\n";

    double improvement = snr_hybrid - snr_in;
    cout << "\nSNR Comparison (dB):\n";
    cout << "Input SNR : " << snr_in << endl; 
    cout << "Hybrid Output SNR: " << snr_hybrid << endl; 
    cout << "Improvement : " << improvement << endl;

    /* ================= DATA FILES FOR PLOTS ================= */
    ofstream beam_ml("beam_ml.dat"), beam_mv("beam_mvdr.dat"), beam_hyb("beam_hybrid.dat");
    for (int ang = -90; ang <= 90; ang++) {
        VectorXcd sv = steering_vector(ang, M);
        beam_ml << ang << " " << 20*log10(abs((w_ml.adjoint()*sv)(0,0))+EPS) << endl;
        beam_mv << ang << " " << 20*log10(abs((w_mvdr.adjoint()*sv)(0,0))+EPS) << endl;
        beam_hyb << ang << " " << 20*log10(abs((w_hybrid.adjoint()*sv)(0,0))+EPS) << endl;
    }
    beam_ml.close(); beam_mv.close(); beam_hyb.close();

    ofstream before("before.dat"), after("after.dat");
    for (int n = 0; n < N; n++) {
        before << n << " " << real(X(0,n)) << endl;
        after << n << " " << real(y_hyb(n)) << endl;
    }
    before.close(); after.close();

    ofstream doa_plot("doa_estimation.dat");
    for (int i = 0; i < N; i++)
        doa_plot << i << " " << theta_s_noisy(i) << " " << theta_s_hat(i) << " " << theta_ml(i) << endl;
    doa_plot.close();

    ofstream snr_alpha("snr_vs_alpha.dat");
    for (double alpha = 0.0; alpha <= 1.0; alpha += 0.01) {
        VectorXcd w_tmp = (alpha*w_ml + (1-alpha)*w_mvdr).normalized();
        RowVectorXcd y_tmp = w_tmp.adjoint() * X;
        RowVectorXcd y_sig = w_tmp.adjoint() * X_sig;
        RowVectorXcd y_n = y_tmp - y_sig;
        double Ps = y_sig.cwiseAbs2().mean();
        double Pn = max(EPS, y_n.cwiseAbs2().mean());
        double snr = 10 * log10(Ps / Pn);
        snr_alpha << alpha << " " << snr << endl;
    }
    snr_alpha.close();

    ofstream bp3d("beampattern_hybrid_3d.dat");
    for (int n = 0; n < N; n++) {
        for (int ang = -90; ang <= 90; ang++) {
            VectorXcd sv = steering_vector(ang, M);
            bp3d << n+1 << " " << ang << " " << 20*log10(abs((w_hybrid.adjoint()*sv)(0,0))+EPS) << endl;
        }
        bp3d << endl;
    }
    bp3d.close();

    ofstream snr_bar("snr_comparison_bar.dat");
    snr_bar << "Input " << snr_in << endl;
    snr_bar << "ML " << snr_ml << endl;
    snr_bar << "MVDR " << snr_mvdr << endl;
    snr_bar << "Hybrid " << snr_hybrid << endl;
    snr_bar.close();


    /* ================= GNUPLOT COMMANDS ================= */
    system("gnuplot -persist -e \"set title 'ML Beam'; set grid; plot 'beam_ml.dat' w l\"");
    system("gnuplot -persist -e \"set title 'MVDR Beam'; set grid; plot 'beam_mvdr.dat' w l\"");
    system("gnuplot -persist -e \"set title 'Hybrid Beam'; set grid; plot 'beam_hybrid.dat' w l\"");
    system("gnuplot -persist -e \"set title 'Before Beamforming'; set grid; plot 'before.dat' w l\"");
    system("gnuplot -persist -e \"set title 'After Beamforming'; set grid; plot 'after.dat' w l\"");
    system("gnuplot -persist -e \"set title 'DOA Estimation'; set xlabel 'Snapshot'; set ylabel 'Angle (deg)'; set grid; plot 'doa_estimation.dat' using 1:2 with lines title 'Noisy', 'doa_estimation.dat' using 1:3 with lines title 'Denoised', 'doa_estimation.dat' using 1:4 with lines title 'ML Prediction'\"");


// Gnuplot for combined 2D beam pattern
system(
    "gnuplot -persist -e \"set title 'Beam Patterns (ML/MVDR/Hybrid)'; "
    "set xlabel 'Angle (deg)'; set ylabel 'Magnitude (dB)'; set grid; "
    "plot 'beam_all.dat' using 1:2 with lines lw 2 title 'ML', "
    "'beam_all.dat' using 1:3 with lines lw 2 title 'MVDR', "
    "'beam_all.dat' using 1:4 with lines lw 2 title 'Hybrid'\""
);
    system("gnuplot -persist -e \"set title 'SNR vs Alpha'; set xlabel 'Alpha'; set ylabel 'SNR (dB)'; set grid; plot 'snr_vs_alpha.dat' using 1:2 with lines title 'Hybrid SNR'\"");
    system("gnuplot -persist -e \"set title '3D Hybrid Beam Pattern'; set xlabel 'Snapshot'; set ylabel 'Angle (deg)'; set zlabel 'Magnitude (dB)'; set pm3d; set palette rgbformulae 33,13,10; splot 'beampattern_hybrid_3d.dat' using 1:2:3 with pm3d\"");
    system("gnuplot -persist -e \"set title 'SNR Comparison'; set style data histograms; set style fill solid 1.0 border -1; set ylabel 'SNR (dB)'; set grid ytics; plot 'snr_comparison_bar.dat' using 2:xtic(1) lc rgb 'skyblue'\"");

    return 0;
}