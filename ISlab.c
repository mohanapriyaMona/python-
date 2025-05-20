//// Playfair Substitution technique
#include <stdio.h>
#include <string.h>
#include <ctype.h>

#define MX 5

// Encrypts a pair using the Playfair cipher
void playfair(char a, char b, char key[MX][MX]) {
    int row1, col1, row2, col2;
    FILE *out = fopen("chiper.txt", "a+");

    // Find positions of a and b in the key matrix
    for (int i = 0; i < MX; i++) {
        for (int j = 0; j < MX; j++) {
            if (key[i][j] == a) { row1 = i; col1 = j; }
            if (key[i][j] == b) { row2 = i; col2 = j; }
        }
    }

    // Apply Playfair cipher rules
    if (row1 == row2) {
        col1 = (col1 + 1) % MX;
        col2 = (col2 + 1) % MX;
    } else if (col1 == col2) {
        row1 = (row1 + 1) % MX;
        row2 = (row2 + 1) % MX;
    } else {
        int temp = col1;
        col1 = col2;
        col2 = temp;
    }

    printf("%c%c", key[row1][col1], key[row2][col2]);
    fprintf(out, "%c%c", key[row1][col1], key[row2][col2]);
    fclose(out);
}

int main() {
    char key[MX][MX], used[26] = {0}, str[100], keystr[26];
    int i, j, k = 0;

    // Get key
    printf("Enter key: ");
    fgets(keystr, sizeof(keystr), stdin);
    keystr[strcspn(keystr, "\n")] = '\0';

    // Get plain text
    printf("Enter plain text: ");
    fgets(str, sizeof(str), stdin);
    str[strcspn(str, "\n")] = '\0';

    // Prepare key matrix (convert to uppercase, replace J with I, remove duplicates)
    for (i = 0; keystr[i]; i++) {
        char ch = toupper(keystr[i]);
        if (ch == 'J') ch = 'I';
        if (ch >= 'A' && ch <= 'Z' && !used[ch - 'A']) {
            used[ch - 'A'] = 1;
            key[k / MX][k % MX] = ch;
            k++;
        }
    }

    // Fill remaining letters
    for (i = 0; i < 26; i++) {
        if (i == 'J' - 'A') continue;
        if (!used[i]) {
            key[k / MX][k % MX] = 'A' + i;
            k++;
        }
    }

    // Display key matrix
    printf("\nKey Matrix:\n");
    for (i = 0; i < MX; i++) {
        for (j = 0; j < MX; j++)
            printf("%c ", key[i][j]);
        printf("\n");
    }

    // Convert plaintext to uppercase and replace J with I
    int len = strlen(str);
    for (i = 0; i < len; i++) {
        str[i] = toupper(str[i]);
        if (str[i] == 'J') str[i] = 'I';
    }

    // Encrypt plaintext
    printf("\nCipher Text: ");
    for (i = 0; i < len; i++) {
        char a = str[i];
        if (!isalpha(a)) continue;
        a = toupper(a);
        if (a == 'J') a = 'I';

        char b = 'X';
        int j = i + 1;
        while (j < len && !isalpha(str[j])) j++;

        if (j < len) {
            b = toupper(str[j]);
            if (b == 'J') b = 'I';
            if (a == b) {
                b = 'X';
            } else {
                i = j; // skip next character
            }
        }

        playfair(a, b, key);
    }

    printf("\n");
    return 0;
}





#include <stdio.h>
#include <conio.h>
#include <string.h>

int main() {
    unsigned int a[3][3] = {
        {6, 24, 1},
        {13, 16, 10},
        {20, 17, 15}
    };

    unsigned int b[3][3] = {
        {8, 5, 10},
        {21, 8, 21},
        {21, 12, 8}
    };

    int i, j, t = 0;
    unsigned int c[20], d[20];
    char msg[20];

    clrscr();

    printf("Enter plain text: ");
    scanf("%s", msg);

    // Convert characters to numerical values (A=0, B=1, ..., Z=25)
    for (i = 0; i < strlen(msg); i++) {
        c[i] = msg[i] - 65;
        printf("%d ", c[i]);
    }

    // Encryption: multiply with key matrix 'a' and mod 26
    for (i = 0; i < 3; i++) {
        t = 0;
        for (j = 0; j < 3; j++) {
            t += a[i][j] * c[j];
        }
        d[i] = t % 26;
    }

    printf("\nEncrypted Cipher Text: ");
    for (i = 0; i < 3; i++) {
        printf("%c ", d[i] + 65);
    }

    // Decryption: multiply with inverse matrix 'b' and mod 26
    for (i = 0; i < 3; i++) {
        t = 0;
        for (j = 0; j < 3; j++) {
            t += b[i][j] * d[j];
        }
        c[i] = t % 26;
    }

    printf("\nDecrypted Cipher Text: ");
    for (i = 0; i < 3; i++) {
        printf("%c ", c[i] + 65);
    }

    getch();
    return 0;
}





#include <stdio.h>
#include <string.h>
#include <conio.h>
#include <ctype.h>

int main() {
    char plain[10], cipher[10];
    int key, i, length;

    clrscr();

    printf("\nEnter the plain text: ");
    scanf("%s", plain);

    printf("\nEnter the key value: ");
    scanf("%d", &key);

    printf("\n\n\tPLAIN TEXT: %s", plain);
    printf("\n\n\tENCRYPTED TEXT: ");

    // Encryption process
    for (i = 0, length = strlen(plain); i < length; i++) {
        cipher[i] = plain[i] + key;

        // Wrap around for uppercase letters
        if (isupper(plain[i]) && (cipher[i] > 'Z'))
            cipher[i] -= 26;

        // Wrap around for lowercase letters
        if (islower(plain[i]) && (cipher[i] > 'z'))
            cipher[i] -= 26;

        printf("%c", cipher[i]);
    }

    printf("\n\n\tAFTER DECRYPTION: ");

    // Decryption process
    for (i = 0; i < length; i++) {
        plain[i] = cipher[i] - key;

        // Wrap around for uppercase letters
        if (isupper(cipher[i]) && (plain[i] < 'A'))
            plain[i] += 26;

        // Wrap around for lowercase letters
        if (islower(cipher[i]) && (plain[i] < 'a'))
            plain[i] += 26;

        printf("%c", plain[i]);
    }

    getch();
    return 0;
}
