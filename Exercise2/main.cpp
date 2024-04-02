#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

// Funzione per errore relativo
double erroreRelativo(const VectorXd& x, const VectorXd& x_exact) {
    double norm_x_exact = x_exact.norm();
    return (x - x_exact).norm() / norm_x_exact;
}

int main() {
    // Definisco le matrici e i vettori del tre sistemi
    Matrix2d A1, A2, A3;
    Vector2d b1, b2, b3;

    A1 << 5.547001962252291e-01, -3.770900990025203e-02,8.320502943378437e-01, -9.992887623566787e-01;
    b1 << -5.169911863249772e-01, 1.672384680188350e-01;

    A2 << 5.547001962252291e-01, -5.540607316466765e-01,8.320502943378437e-01, -8.324762492991313e-01;
    b2 << -6.394645785530173e-04, 4.259549612877223e-04;

    A3 << 5.547001962252291e-01, -5.547001955851905e-01,8.320502943378437e-01, -8.320502947645361e-01;
    b3 << -6.400391328043042e-10, 4.266924591433963e-10;

    // soluzioni esatte dei sistemi
    VectorXd x_esatto(2);
    x_esatto << -1.0e+0, -1.0e+00;

    // risoluzione con metodo PALU
    VectorXd x1_palu, x2_palu, x3_palu;
    x1_palu = A1.partialPivLu().solve(b1);
    x2_palu = A2.partialPivLu().solve(b2);
    x3_palu = A3.partialPivLu().solve(b3);

    // risoluzione con metodo QR
    VectorXd x1_qr, x2_qr, x3_qr;
    x1_qr = A1.householderQr().solve(b1);
    x2_qr = A2.householderQr().solve(b2);
    x3_qr = A3.householderQr().solve(b3);

    // calcolo errori relativi
    double rel_error1_palu = erroreRelativo(x1_palu, x_esatto);
    double rel_error2_palu = erroreRelativo(x2_palu, x_esatto);
    double rel_error3_palu = erroreRelativo(x3_palu, x_esatto);

    double rel_error1_qr = erroreRelativo(x1_qr, x_esatto);
    double rel_error2_qr = erroreRelativo(x2_qr, x_esatto);
    double rel_error3_qr = erroreRelativo(x3_qr, x_esatto);

    // Stampo risultati
    cout << "Decomposizione PALU:" << endl;
    cout << "\t" << "Errore relativo del sistema 1: " << rel_error1_palu << endl;
    cout <<"\t" << "Errore relativo del sistema 2: " << rel_error2_palu << endl;
    cout <<"\t" << "Errore relativo del sistema 3: " << rel_error3_palu << endl;

    cout <<"Decomposizione QR:" << endl;
    cout <<"\t" << "Errore relativo del sistema 1: " << rel_error1_qr << endl;
    cout << "\t" <<"Errore relativo del sistema 2: " << rel_error2_qr << endl;
    cout << "\t" <<"Errore relativo del sistema 3: " << rel_error3_qr << endl;

    return 0;
}
