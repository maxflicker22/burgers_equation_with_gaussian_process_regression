def generated_function_dkdxdydlogtheta(logsigma, logtheta, x, y):
    return -3*ehoch1**logtheta*x**2*(2*ehoch1**logtheta + 4*ehoch1**(logsigma + logtheta))/(2*pi*(-(ehoch1**logsigma + ehoch1**logtheta*x*y)**2/((ehoch1**logsigma + ehoch1**logtheta*x**2 + 1)*(ehoch1**logsigma + ehoch1**logtheta*y**2 + 1)) + 1)**(3/2)*(ehoch1**logsigma + ehoch1**logtheta*x**2 + 1)**(5/2)*(ehoch1**logsigma + ehoch1**logtheta*y**2 + 1)**(3/2)) - 3*ehoch1**logtheta*y**2*(2*ehoch1**logtheta + 4*ehoch1**(logsigma + logtheta))/(2*pi*(-(ehoch1**logsigma + ehoch1**logtheta*x*y)**2/((ehoch1**logsigma + ehoch1**logtheta*x**2 + 1)*(ehoch1**logsigma + ehoch1**logtheta*y**2 + 1)) + 1)**(3/2)*(ehoch1**logsigma + ehoch1**logtheta*x**2 + 1)**(3/2)*(ehoch1**logsigma + ehoch1**logtheta*y**2 + 1)**(5/2)) + (2*ehoch1**logtheta + 4*ehoch1**(logsigma + logtheta))/(pi*(-(ehoch1**logsigma + ehoch1**logtheta*x*y)**2/((ehoch1**logsigma + ehoch1**logtheta*x**2 + 1)*(ehoch1**logsigma + ehoch1**logtheta*y**2 + 1)) + 1)**(3/2)*(ehoch1**logsigma + ehoch1**logtheta*x**2 + 1)**(3/2)*(ehoch1**logsigma + ehoch1**logtheta*y**2 + 1)**(3/2)) - 3*(2*ehoch1**logtheta + 4*ehoch1**(logsigma + logtheta))*(ehoch1**logtheta*x**2*(ehoch1**logsigma + ehoch1**logtheta*x*y)**2/((ehoch1**logsigma + ehoch1**logtheta*x**2 + 1)**2*(ehoch1**logsigma + ehoch1**logtheta*y**2 + 1)) - 2*ehoch1**logtheta*x*y*(ehoch1**logsigma + ehoch1**logtheta*x*y)/((ehoch1**logsigma + ehoch1**logtheta*x**2 + 1)*(ehoch1**logsigma + ehoch1**logtheta*y**2 + 1)) + ehoch1**logtheta*y**2*(ehoch1**logsigma + ehoch1**logtheta*x*y)**2/((ehoch1**logsigma + ehoch1**logtheta*x**2 + 1)*(ehoch1**logsigma + ehoch1**logtheta*y**2 + 1)**2))/(2*pi*(-(ehoch1**logsigma + ehoch1**logtheta*x*y)**2/((ehoch1**logsigma + ehoch1**logtheta*x**2 + 1)*(ehoch1**logsigma + ehoch1**logtheta*y**2 + 1)) + 1)**(5/2)*(ehoch1**logsigma + ehoch1**logtheta*x**2 + 1)**(3/2)*(ehoch1**logsigma + ehoch1**logtheta*y**2 + 1)**(3/2))
